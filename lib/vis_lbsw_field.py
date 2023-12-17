from PIL import Image
import numpy as np

body_model = 'smplx'

def scalar2saturation(scalars, hue_value = 120, clip_min = 0.0, clip_max = 1.0):
        image = Image.new("HSV",(scalars.shape[0], 1))
        np_im = 0.0*np.array(image)
        scalars = np.clip(scalars, clip_min, clip_max)
        # hue
        np_im[:,:,0] += hue_value*255/360
        # scalar2saturation
        np_im[:,:,1] += 255*scalars
        # value
        np_im[:,:,2] += 255
        rgb_image = Image.fromarray(np_im.astype(np.uint8), mode="HSV").convert(mode="RGB")

        return np.array(rgb_image)[0].astype(np.float64)/255.0


if body_model == 'smpl':
    joint_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    previous_joint = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    # unique_parent_joint = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    hue_colors = np.arange(start=0, stop=360, step=360/(len(joint_list)))
else:
    joint_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    previous_joint = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    # unique_parent_joint = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    hue_colors = np.arange(start=0, stop=360, step=360/(len(joint_list)))

## Step 1: Create 2D planar grid that corresponds to image pixel grid ##
xy_plane_points = None
linspaces = [np.linspace(0.0, 512.0, num=512) for d in range(0, 2)]
coords = np.meshgrid(*linspaces,indexing='ij') 
xy_plane_points = np.concatenate([coords[i].reshape([-1, 1]) for i in range(0, 2)], axis=1)
xy_plane_points = np.hstack((xy_plane_points, 0*xy_plane_points[:,:1]))
xy_plane_points = max_size/480*xy_plane_points

current_center = np.array([max_size/480*255.5, max_size/480*255.5, 0.0])
xy_plane_points = xy_plane_points - current_center + xy_plane_origin
xy_plane_points = torch.Tensor(xy_plane_points)[None,:,:].float().permute(0,2,1).to(device=cuda)


## Step 2: Query the lbsw of 2D planar gird points ##
pred_lbs_w = inv_skin_net.query(xy_plane_points, bmin=bmin[:,:,None], bmax=bmax[:,:,None])

lbs_image = pred_lbs_w.permute(0,2,1)[0].cpu().numpy()
# lbs_image shape: (WxH, N_joints)

## Step 3: Iterate all joints and accumulate the importance color map into a single RGB image ##
cumulated_rgb_lbs_image = None
for i in range(len(joint_list)):
    rgb_lbs_image = scalar2saturation(lbs_image[:,joint_list[i]], hue_value=hue_colors[i])
    rgb_lbs_image = rgb_lbs_image[None,:,:].reshape(512, 512, 3)

    cumulated_rgb_lbs_image = 1.0 - rgb_lbs_image if cumulated_rgb_lbs_image is None else cumulated_rgb_lbs_image + (1.0-rgb_lbs_image)

cumulated_rgb_lbs_image = np.clip(cumulated_rgb_lbs_image, 0.0, 1.0)
cumulated_rgb_lbs_image = 1.0 - cumulated_rgb_lbs_image
cumulated_rgb_lbs_image = (255*cumulated_rgb_lbs_image).astype(np.uint8)

cumulated_rgb_lbs_image = Image.fromarray(cumulated_rgb_lbs_image)
cumulated_rgb_lbs_image = cumulated_rgb_lbs_image.rotate(90)

cumulated_rgb_lbs_image.save('%s/i%s_jall_e%s.png' % (result_dir, str(idx).zfill(4), name))