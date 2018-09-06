# Deep-Restore

Analog movies, recorded on roll film, have the property that their image quality worsens statically either from usage or by time. To prevent this deterioration from happening, analog movies often get digitalized. The digitalization of analog movies has the advantage that during post-processing steps the image quality of previously analog movies can be improved - this is known as film restoration.

In this theses, we will focus on such a post-processing step, namely the removing of impurities in single image frames. There are different types of impurities and they can by categorized into: dust, scratches, flicker and noise. 

We split this problem into parts: first, we detect the impurities of each frame and generate a segmentation map (image segmentation), and second, based on this segmentation map we fill the impurities by using information from surrounding pixels (image in-painting).

For the image segmentation we use a modern machine learning approach, where we focus on convolutional neural networks (CNNs). We test different architectures and regularization techniques to come up with a suitable architecture which gives the best results for the segmentation map.

After we generated the segmentation map we can perform image in-painting of the impure pixels. Again, we test different methods such as image in-painting based on Perona-Malik-Diffusion, etc.
