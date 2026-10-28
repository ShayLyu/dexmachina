"""DemoMimic-style contact rewards: alignment reward (AR) and sustained contact reward (SCR).

AR: for each hand collision link, a fixed pad normal (link frame) should point against the object's
outward surface normal, both at the current contact (if in contact) and at the link's next demo
contact location. The current object normal is the simulator contact normal (force-weighted over the
link pair's contacts), oriented with the hand link's mesh centroid so it points out of the object. r = clamp((cos - s) / (1 - s), 0, 1), summed over the two terms and normalized by
2 * number of links.
SCR: per-hand demo contact windows; inside a window of length S the reward is c(c+1) / (S(S+1)) where c
is the current streak of steps with contact force above a threshold. Averaged over active windows.

Per-clip quantities (pad normals, future demo contacts, windows, object part meshes) come from a cache
built by tools/build_contact_cache.py.
"""
import numpy as np
import torch

from dexmachina.envs.contacts import get_expanded_mask
from dexmachina.envs.math_utils import matrix_from_quat


def get_contact_shaping_cfg():
    return {
        "ar_weight": 0.0,
        "scr_weight": 0.0,
        "ar_scale": 0.5, # s in DemoMimic: cosine similarity below s gets zero reward
        "contact_force_thres": 1.0, # N, minimum force for a link to count as in contact
        "arm_vel_penalty": 0.0,
        "finger_vel_penalty": 0.0,
        "cache_path": None,
    }


class ContactShaping:
    def __init__(self, cfg, robots, obj, num_envs, device, contact_link_slices):
        self.cfg = cfg
        self.robots = robots
        self.object = obj
        self.num_envs = num_envs
        self.device = device
        self.contact_link_slices = contact_link_slices
        self.ar_weight = cfg["ar_weight"]
        self.scr_weight = cfg["scr_weight"]
        self.ar_scale = cfg["ar_scale"]
        self.force_thres = cfg["contact_force_thres"]
        self.arm_vel_penalty = cfg.get("arm_vel_penalty", 0.0)
        self.finger_vel_penalty = cfg.get("finger_vel_penalty", 0.0)
        self.sides = list(robots.keys())

        assert cfg["cache_path"] is not None, "contact shaping needs a cache, see tools/build_contact_cache.py"
        cache = np.load(cfg["cache_path"], allow_pickle=True).item()
        assert cache["obj_link_names"] == obj.link_names, f"{cache['obj_link_names']} vs {obj.link_names}"
        self.T = cache["num_frames"]
        t = lambda x, dtype=torch.float32: torch.tensor(np.asarray(x), dtype=dtype, device=device)
        self.obj_link_idxs = torch.tensor(obj.coll_idxs_global, device=device)

        self.pad_normals, self.link_centroids, self.future_part, self.future_normal = dict(), dict(), dict(), dict()
        self.window_id, self.window_len, self.robot_link_idxs = dict(), dict(), dict()
        self.streak, self.current_window = dict(), dict()
        self.arm_idxs, self.finger_idxs = dict(), dict()
        for side, robot in robots.items():
            c = cache["sides"][side]
            assert c["link_names"] == robot.coll_link_names, f"{side}: {c['link_names']} vs {robot.coll_link_names}"
            self.pad_normals[side] = t(c["pad_normals"]) # (L, 3) link frame
            self.link_centroids[side] = t(c["link_centroids"]) # (L, 3) link frame
            self.robot_link_idxs[side] = torch.tensor(robot.coll_idxs_global, device=device)
            self.future_part[side] = t(c["future_part"], torch.long) # (T, L), -1 if no future contact
            self.future_normal[side] = t(c["future_normal"]) # (T, L, 3) object link frame
            self.window_id[side] = t(c["window_id"], torch.long) # (T,), -1 outside windows
            self.window_len[side] = t(c["window_len"] if len(c["window_len"]) else [1.0]) # (num_windows,)
            self.streak[side] = torch.zeros(num_envs, device=device)
            self.current_window[side] = torch.full((num_envs,), -1, dtype=torch.long, device=device)
            names = robot.actuated_dof_names
            self.arm_idxs[side] = [i for i, n in enumerate(names) if 'forearm' in n]
            self.finger_idxs[side] = [i for i, n in enumerate(names) if 'forearm' not in n]

    def reset_idx(self, env_idxs):
        for side in self.sides:
            self.streak[side][env_idxs] = 0.0
            self.current_window[side][env_idxs] = -1

    def contact_normals(self, side):
        """Force-weighted sum of simulator contact normals per (object link, hand link): (N, n_obj_links, L, 3).
        The sign is consistent within a link pair but arbitrary, see compute()."""
        collider = self.object.entity._solver.collider
        data = collider.contact_data
        n_contacts = collider.n_contacts.to_torch(device=self.device)
        link_a = data.link_a.to_torch(device=self.device)
        link_b = data.link_b.to_torch(device=self.device)
        normal = data.normal.to_torch(device=self.device) # (n_max, N, 3)
        weight = torch.norm(data.force.to_torch(device=self.device), dim=-1, keepdim=True)
        mask = get_expanded_mask(n_contacts, link_a, link_b, self.obj_link_idxs, self.robot_link_idxs[side])
        return torch.einsum("cnab,cnk->nabk", mask.float(), normal * weight)

    def scaled_alignment(self, pad_world, obj_normal):
        cos = -(pad_world * obj_normal).sum(dim=-1) # pad should push against the outward normal
        return torch.clamp((cos - self.ar_scale) / (1.0 - self.ar_scale), 0.0, 1.0)

    def compute(self, contact_link_pos, contact_link_valid, contact_forces, episode_length_buf):
        """contact_link_pos: (N, n_obj_links, n_robot_links, 3); contact_forces: (N, n_obj_links, n_robot_links, 3)"""
        t = torch.clamp(episode_length_buf.long(), max=self.T - 1)
        rew = torch.zeros(self.num_envs, device=self.device)
        info = dict()
        ar_sum, n_links = 0.0, 0
        scr_sum, n_active = 0.0, 0.0
        for side, robot in self.robots.items():
            sl = self.contact_link_slices[side]
            force = torch.norm(contact_forces[:, :, sl], dim=-1) # (N, n_obj_links, L)
            in_contact = contact_link_valid[:, :, sl] & (force > self.force_thres)
            link_quat = robot.entity.get_links_quat()[:, robot.coll_idxs_local] # (N, L, 4)
            N, L = link_quat.shape[:2]
            link_rot = matrix_from_quat(link_quat.reshape(-1, 4)).reshape(N, L, 3, 3)
            pad_world = torch.einsum("nlij,lj->nli", link_rot, self.pad_normals[side])
            if self.ar_weight > 0:
                # current term: the object part with the larger contact force for each link
                part = force.argmax(dim=1) # (N, L)
                pick = lambda x: torch.gather(x, 1, part[:, None, :, None].expand(-1, 1, -1, 3))[:, 0]
                pos = pick(contact_link_pos[:, :, sl])
                cur_normal = torch.nn.functional.normalize(pick(self.contact_normals(side)), dim=-1)
                # orient out of the object: from the contact point toward the hand link's centroid
                link_pos = robot.entity.get_links_pos()[:, robot.coll_idxs_local]
                centroid = link_pos + torch.einsum("nlij,lj->nli", link_rot, self.link_centroids[side])
                flip = ((centroid - pos) * cur_normal).sum(dim=-1, keepdim=True) < 0
                cur_normal = torch.where(flip, -cur_normal, cur_normal)
                r_cur = self.scaled_alignment(pad_world, cur_normal) * in_contact.any(dim=1).float()
                # future term: object normal at the next demo contact, in the part's current pose
                fpart = self.future_part[side][t] # (N, L)
                fnormal_local = self.future_normal[side][t] # (N, L, 3)
                rot = matrix_from_quat(self.object.part_quat.reshape(-1, 4)).reshape(N, -1, 3, 3)
                rot = torch.gather(rot, 1, fpart.clamp(min=0)[..., None, None].expand(-1, -1, 3, 3))
                fnormal = torch.einsum("nlij,nlj->nli", rot, fnormal_local)
                r_fut = self.scaled_alignment(pad_world, fnormal) * (fpart >= 0).float()
                ar_sum = ar_sum + (r_cur + r_fut).sum(dim=-1)
                n_links += L
                info[f"ar_cur_{side}"] = r_cur.mean(dim=-1)
                info[f"ar_fut_{side}"] = r_fut.mean(dim=-1)
                info[f"n_contact_links_{side}"] = in_contact.any(dim=1).float().sum(dim=-1)
            if self.scr_weight > 0:
                wid = self.window_id[side][t] # (N,)
                new_window = wid != self.current_window[side]
                self.streak[side][new_window] = 0.0
                self.current_window[side][:] = wid
                active = wid >= 0
                touching = in_contact.any(dim=(1, 2))
                self.streak[side] = torch.where(active & touching, self.streak[side] + 1, torch.zeros_like(self.streak[side]))
                S = self.window_len[side][wid.clamp(min=0)]
                c = self.streak[side]
                r_side = torch.where(active, c * (c + 1) / (S * (S + 1)), torch.zeros_like(c))
                scr_sum = scr_sum + r_side
                n_active = n_active + active.float()
                info[f"scr_streak_{side}"] = c
        if self.ar_weight > 0:
            ar = ar_sum / (2.0 * n_links)
            rew += self.ar_weight * ar
            info["ar"] = ar
            info["ar_rew"] = self.ar_weight * ar
        if self.scr_weight > 0:
            scr = scr_sum / torch.clamp(n_active, min=1.0)
            rew += self.scr_weight * scr
            info["scr"] = scr
            info["scr_rew"] = self.scr_weight * scr
        if self.arm_vel_penalty > 0 or self.finger_vel_penalty > 0:
            pen = 0.0
            for side, robot in self.robots.items():
                vel = robot.dof_vel
                pen = pen + self.arm_vel_penalty * (vel[:, self.arm_idxs[side]] ** 2).mean(dim=-1)
                pen = pen + self.finger_vel_penalty * (vel[:, self.finger_idxs[side]] ** 2).mean(dim=-1)
            pen = pen / len(self.robots)
            rew -= pen
            info["vel_penalty"] = pen
        return rew, info
