import os
import sys
import time
from pathlib import Path

CWD = Path(__file__).resolve()
sys.path.append(CWD.parent.parent.as_posix())
sys.path.append(CWD.parent.parent.parent.as_posix())

import cv2
import faiss
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import DataParallel
from sklearn.metrics import accuracy_score

from const import CelebVNID, CONFIG_PATH, CONFIG_NAME
from arcface.model import resnet_face18


def verify_device(device):
    if device != 'cpu' and torch.cuda.is_available():
        return device
    return 'cpu'


def name_format(celeb):
    return celeb.replace(' ', '_')


class ArcFaceInference:
    def __init__(self, device, cfg):
        self.device = verify_device(device)
        self.img_size = cfg.img_size
        self.use_se = cfg.use_se
        self.use_faiss = cfg.use_faiss
        self.model = self.load_model(weight_path=cfg.weight_path)

        self.threshold = cfg.threshold

        if os.path.exists(cfg.mask_path):
            self.mask_coresets = torch.load(cfg.mask_path)
        else:
            self.mask_coresets = None

        if os.path.exists(cfg.face_path):
            self.face_coresets = torch.load(cfg.face_path)
        else:
            self.face_coresets = None

        if self.use_faiss:
            if os.path.exists(cfg.mask_faiss):
                self.mask_faiss_coresets = torch.load(cfg.mask_faiss)
            else:
                self.mask_faiss_coresets = None

            if os.path.exists(cfg.face_faiss):
                self.face_faiss_coresets = torch.load(cfg.face_faiss)
            else:
                self.face_faiss_coresets = None
            self.face_index, self.mask_index = self.create_nn_index()

    def load_model(self, weight_path):
        model = resnet_face18(use_se=self.use_se)
        model = DataParallel(model)
        model_state = torch.load(weight_path, map_location=self.device)

        try:
            model.load_state_dict(model_state)
        except:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in model_state.items():
                name = k.replace('module.', '', 1)
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        model.to(self.device)
        model.eval()
        return model

    def predict(self, input, mask=False, coresets=False):
        input = self._preprocess(input)

        output = self.model(input)

        output = output.detach().numpy()
        feature_1 = output[::2]
        feature_2 = output[1::2]
        feature = np.hstack((feature_1, feature_2))

        if coresets:
            return feature

        label = self._postprocess(feature, mask)
        return label

    def _preprocess(self, input):
        if isinstance(input, str):  # path
            input = cv2.imread(input, 0)
        elif isinstance(input, np.ndarray):
            input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

        img_resize = cv2.resize(input, (self.img_size))
        img_lst = np.dstack((img_resize, np.fliplr(img_resize)))
        img_lst = img_lst.transpose((2, 0, 1))
        img_lst = img_lst[:, np.newaxis, :, :]
        image_nor = img_lst.astype(np.float32, copy=False)

        image_nor -= 127.5
        image_nor /= 127.5

        img_tensor = torch.from_numpy(image_nor)
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def _postprocess(self, feature: np.asarray, mask: bool = False):
        min_distance = 100

        if self.use_faiss:
            if mask:
                min_distance, idx = self.mask_index.search(feature, 1)
                name_pred = self.mask_faiss_coresets['target'][int(idx[0])]
                min_distance = float(min_distance[0])
            else:
                min_distance, idx = self.face_index.search(feature, 1)
                name_pred = self.face_faiss_coresets['target'][int(idx[0])]
                min_distance = float(min_distance[0])
        else:
            if mask:
                coresets = self.mask_coresets
            else:
                coresets = self.face_coresets

            for embedding in coresets:
                euclidean_distance = F.pairwise_distance(
                    torch.from_numpy(feature), torch.from_numpy(embedding['feature'])
                )
                if euclidean_distance < min_distance:
                    name_pred = embedding['name']
                    min_distance = euclidean_distance

        if min_distance <= self.threshold:
            # _{round(min_distance, 2)}
            if mask:
                return f"Deo Khau Trang:{name_pred}"
            else:
                return f"Khong Deo Khau Trang:{name_pred}"
        return (
            "Deo Khau Trang: Khong xac dinh" if mask else "Khong Deo Khau Trang: Khong xac dinh"
        )

    def create_nn_index(self, shape=1024):
        if self.face_faiss_coresets is None or self.mask_faiss_coresets is None:
            return None, None
        face_index = faiss.IndexFlatL2(shape)
        face_index.add(self.face_faiss_coresets['coresets'].numpy())

        mask_index = faiss.IndexFlatL2(shape)
        mask_index.add(self.mask_faiss_coresets['coresets'].numpy())

        return face_index, mask_index

    @staticmethod
    def cosin_metric(x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


@hydra.main(config_name=CONFIG_NAME, config_path=CONFIG_PATH, version_base=None)
def create_coresets(cfg):
    cfg_path = {
        'not mask': (cfg.coresets.face_root, cfg.arcface.face_path),
        'mask': (cfg.coresets.mask_root, cfg.arcface.mask_path),
    }

    for name_face in cfg_path.keys():
        root_path, out_path = cfg_path[name_face]
        if os.path.exists(out_path):
            print(f"Coresets for [{name_face}] embedding dataset already exists")
        else:
            print(f"Start creating [{name_face}] coresets...")
            identify = ArcFaceInference(device=cfg.device, cfg=cfg.arcface)

            lst_celebs = sorted(os.listdir(root_path))

            embeddings_list = list()
            for celeb in lst_celebs:
                # id_celeb = CelebVNID[name_format(celeb)].value
                celeb_path = os.path.join(root_path, celeb)

                for file_name in tqdm(
                    sorted(os.listdir(celeb_path)), desc=f"Embdedding for [{celeb}]"
                ):
                    file_path = os.path.join(celeb_path, file_name)
                    img = cv2.imread(file_path)

                    feature = identify.predict(img, coresets=True)

                    embeddings_list.append(
                        {'feature': feature, 'name': name_format(celeb)}
                    )
            torch.save(embeddings_list, out_path)
            print(f"Finished creating [{name_face}] coresets => out path: [{out_path}]")


@hydra.main(config_name=CONFIG_NAME, config_path=CONFIG_PATH, version_base=None)
def create_faiss_coresets(cfg):
    cfg_path = {
        'not mask': (cfg.coresets.face_root, cfg.arcface.face_faiss),
        'mask': (cfg.coresets.mask_root, cfg.arcface.mask_faiss),
    }

    for name_face in cfg_path.keys():
        root_path, out_path = cfg_path[name_face]
        if os.path.exists(out_path):
            print(f"Coresets for [{name_face}] embedding dataset already exists")
        else:
            print(f"Start creating [{name_face}] coresets...")
            identify = ArcFaceInference(device=cfg.device, cfg=cfg.arcface)

            lst_celebs = sorted(os.listdir(root_path))

            id_lists = dict()
            idx = 0
            for celeb in lst_celebs:
                # id_celeb = CelebVNID[name_format(celeb)].value
                celeb_path = os.path.join(root_path, celeb)

                for file_name in tqdm(
                    sorted(os.listdir(celeb_path)), desc=f"Embdedding for [{celeb}]"
                ):
                    file_path = os.path.join(celeb_path, file_name)
                    img = cv2.imread(file_path)

                    feature = identify.predict(img, coresets=True)

                    id_lists[idx] = name_format(celeb)
                    if idx == 0:
                        coresets = torch.from_numpy(feature)
                    else:
                        coresets = torch.cat(
                            (coresets, torch.from_numpy(feature)),
                            dim=0,
                        )
                    idx += 1

            torch.save({'target': id_lists,
                        'coresets': coresets}, out_path)
            print(f"Finished creating [{name_face}] coresets => out path: [{out_path}]")


@hydra.main(config_name=CONFIG_NAME, config_path=CONFIG_PATH, version_base=None)
def evaluate(cfg):
    identify = ArcFaceInference(device=cfg.device, cfg=cfg.arcface)

    lst_celebs = sorted(os.listdir(cfg.root_test))

    targets, predictions = [], []
    inf_time = 0
    num_samples = len(lst_celebs)
    for celeb in lst_celebs:
        id_celeb = CelebVNID[name_format(celeb)].value

        celeb_path = os.path.join(cfg.root_test, celeb)
        for file_name in tqdm(
            sorted(os.listdir(celeb_path)), desc=f"Testing for [{celeb}]"
        ):
            file_path = os.path.join(celeb_path, file_name)
            img = cv2.imread(file_path)
            start_time = time.perf_counter()
            id_pred, _ = identify.predict(img, mask=False)
            inf_time += time.perf_counter() - start_time
            targets.append(id_celeb)
            predictions.append(id_pred)

    # accuracy = accuracy_score(targets, predictions)
    # print(f"Accuracy: {round(accuracy * 100, 2)}%")
    print(f"Time inf: {inf_time / num_samples}")


@hydra.main(config_name=CONFIG_NAME, config_path=CONFIG_PATH, version_base=None)
def make_id(cfg):
    lst_celebs = sorted(os.listdir(cfg.coresets.face_root))

    for idx, celeb in enumerate(sorted(lst_celebs)):
        print(f"{name_format(celeb)} = {idx}")


@hydra.main(config_name=CONFIG_NAME, config_path=CONFIG_PATH, version_base=None)
def normalize_image_names(cfg):
    lst_roots = [cfg.normalize.root]

    for root_path in lst_roots:
        print(f"Processing for [{root_path}]")
        name_folder = root_path.split("/")[-1]
        new_folder = f"new_{name_folder}"
        new_root_path = '/'.join(root_path.split("/")[:-1])

        for celeb in sorted(os.listdir(root_path)):
            origin_path = os.path.join(root_path, celeb)
            new_path = os.path.join(new_root_path, new_folder, celeb)
            os.makedirs(new_path, exist_ok=True)

            for idx, file_name in tqdm(
                enumerate(sorted(os.listdir(origin_path))),
                desc=f"Formating image names for [{celeb}:]",
            ):
                origin_file = os.path.join(origin_path, file_name)
                new_file = os.path.join(new_path, f"{celeb}__{idx}.jpg")
                os.rename(origin_file, new_file)


if __name__ == '__main__':
    # create_coresets()
    create_faiss_coresets()
    evaluate()
#
    # make_id()
    # normalize_image_names()
