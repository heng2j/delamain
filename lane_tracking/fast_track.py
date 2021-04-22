import torch, cv2, PIL
from lane_tracking.model.model import parsingNet
import scipy.special
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R
from lane_tracking.util.common import image_resize
from lane_tracking.util.carla_util import carla_img_to_array
import numpy as np

torch.backends.cudnn.benchmark = True


def fast_lane_init(algo_name):
    if algo_name == "culane":
        import lane_tracking.configs.culane as algo
    elif algo_name == "tusimple":
        import lane_tracking.configs.tusimple as algo
    net = parsingNet(pretrained = False, backbone=algo.backbone,
                     cls_dim = (algo.griding_num+1,algo.cls_num_per_lane, algo.num_lanes),
                     use_aux=False).cuda() # we dont need auxiliary segmentation in testing
    state_dict = torch.load(algo.model_name, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    net.load_state_dict(compatible_state_dict, strict=False)

    return net, algo


def get_lanes(img, net, algo, pitch=38):
    image_arr = carla_img_to_array(img)
    img_transforms = torchxform()

    H = rotation(pitch)
    warped_img = cv2.warpPerspective(image_arr[:, :, (2, 1, 0)], H, (800, 288))

    # vis = warped_img.copy()
    vis = np.zeros_like(warped_img).astype(np.uint8)
    im = PIL.Image.fromarray(np.array(warped_img))
    input = img_transforms(im)
    input = input.unsqueeze(0)
    with torch.no_grad():
        out = net(input.to('cuda'))
        col_sample = np.linspace(0, 800 - 1, algo.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(algo.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == algo.griding_num] = 0
        out_j = loc

        img_h, img_w = warped_img.shape[:2]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
        for i in range(out_j.shape[1]):
          if np.sum(out_j[:, i] != 0) > 2:
              for k in range(out_j.shape[0]):
                  if out_j[k, i] > 0:
                      ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                             int(img_h * (algo.row_anchor[algo.cls_num_per_lane-1-k]/288)) - 1 )
                      cv2.circle(vis,ppp,3,colors[i],-1)
        vis2 = image_resize(vis, width=1024, height=512)
        blank_image = np.zeros((144, 1024, 3), np.uint8)
        merge_img = np.concatenate((vis2, blank_image), axis=0)
    return merge_img


def rotation(pitch):
    # Carla intrinsic matrix
    K = np.array(
        [
            [1.23607734e03, 0.00000000e00, 5.12000000e02],
            [0.00000000e00, 1.23607734e03, 2.56000000e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    r = R.from_euler('z', pitch, degrees=True)
    rot = r.as_matrix()
    H = K * rot * np.linalg.inv(K)
    return H


def torchxform():
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return img_transforms
