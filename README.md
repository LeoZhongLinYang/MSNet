# MSNet
Computed Tomography (CT) plays a significant role in various applications, including medical imaging and lesion detection. However, prolonged exposure to high doses of radiation can damage human cells, increase data acquisition costs, and elevate the risk of cancer and related
diseases.

As a result, current mainstream research focuses on reducing radiation dose. The most straightforward approach is to reduce the number of scans, but this leads to a decrease in the number of projections. Fewer projections can result in degraded image resolution during CT reconstruction and make the images more susceptible to noise artifacts, potentially causing significant blurring of image details.

To address this issue, this paper proposes a solution based on a deep learning Convolutional Neural Network (CNN) model. We introduce slight modifications to the traditional U-Net architecture and apply adjustments to both the sinogram and the reconstructed volume to reduce the impact of noise artifacts during reconstruction.

In addition to modifying the model, we designed a custom structure for the input and weights, resulting in a loss function that differs from those used in traditional deep learning approaches.
