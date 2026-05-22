import numpy as np
import SimpleITK as sitk


def track_points_with_demons(
    ref_frame,
    current_frame,
    p0,
    levels=(4, 2, 1),
    iterations_per_level=(40, 20),
    std_dev=3.0,
    roi_margin=None,  # int: crop to bbox(points)+margin before registering
    fixed_image=None,  # optional precomputed sitk fixed image
):
    p0_xy = p0.reshape(-1, 2).astype(np.float64)

    fixed = (
        fixed_image
        if fixed_image is not None
        else sitk.GetImageFromArray(ref_frame.astype(np.float32))
    )
    moving = sitk.GetImageFromArray(current_frame.astype(np.float32))

    if roi_margin is not None:
        w, h = fixed.GetSize()
        x0 = max(int(p0_xy[:, 0].min()) - roi_margin, 0)
        y0 = max(int(p0_xy[:, 1].min()) - roi_margin, 0)
        x1 = min(int(p0_xy[:, 0].max()) + roi_margin, w)
        y1 = min(int(p0_xy[:, 1].max()) + roi_margin, h)
        idx, size = [x0, y0], [x1 - x0, y1 - y0]
        fixed = sitk.RegionOfInterest(fixed, size, idx)
        moving = sitk.RegionOfInterest(moving, size, idx)

    dim = fixed.GetDimension()
    field = None
    for shrink, iters in zip(levels, iterations_per_level):
        if shrink > 1:
            f = sitk.Shrink(fixed, [shrink] * dim)
            m = sitk.Shrink(moving, [shrink] * dim)
        else:
            f, m = fixed, moving

        if field is None:
            disp = sitk.Image(f.GetSize(), sitk.sitkVectorFloat64, dim)
            disp.CopyInformation(f)
        else:
            disp = sitk.Resample(field, f)  # coarse -> coarse, no full-res hop

        demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons.SetNumberOfIterations(int(iters))
        demons.SetStandardDeviations(std_dev)
        field = demons.Execute(f, m, disp)

    # field kept at the finest level's resolution; the transform interpolates
    # it in physical space, so upsampling to full size is unnecessary.
    transform = sitk.DisplacementFieldTransform(
        sitk.Cast(field, sitk.sitkVectorFloat64)
    )

    p1 = np.array(
        [transform.TransformPoint((float(x), float(y))) for x, y in p0_xy],
        dtype=np.float32,
    )
    status = np.ones(len(p0_xy), dtype=np.uint8)
    return p1.reshape(-1, 1, 2), status.reshape(-1, 1)
