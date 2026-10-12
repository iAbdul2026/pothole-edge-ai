Smartphone-Based Edge AI for Real-Time Pothole and Road-Defect Detection on Nigerian Roads
COEN872: AI Systems Development, Department of Computer Engineering, Ahmadu Bello University, Zaria (M.Sc. 2025/26)
Case study 6: Edge AI for Real-Time Pothole and Road-Defect Detection
Author: Abdulhamid Musa
Live demo (open in Chrome): https://iabdul2026.github.io/pothole-edge-ai/demo.html
1. Overview
Potholes and surface defects on Nigerian roads develop faster than they are repaired, and reporting still depends on manual complaints and occasional inspection drives. Cloud-based vision systems assume reliable connectivity, which cannot be taken for granted in much of the country. This project therefore studies whether a compressed object detector can run entirely on a low-cost smartphone, inside a web browser, with no installation, no server round trip and no upload of any image.
The central research question is the trade-off between accuracy, latency, energy and model size for a lightweight detector on low-cost edge hardware. Two configurations of YOLOv8n are compared: a 32-bit floating-point model (FP32) and an 8-bit statically quantised model (INT8). A secondary question is the accuracy gap between an international benchmark dataset and locally collected Nigerian road imagery.
2. Repository contents
Path	Description
docs/demo.html	Browser application: image check, live camera detection and benchmark mode
docs/best.tflite	Fine-tuned YOLOv8n (v1), FP32 export (about 12 MB)
docs/best_int8.tflite	Fine-tuned YOLOv8n (v1), INT8 export (about 3.3 MB)
docs/images/	Example detections used in this README
COEN872_pothole_yolov8_colab.ipynb	Training, evaluation and export notebook (Google Colab, T4 GPU)
The docs folder is served through GitHub Pages, which provides the HTTPS origin that mobile browsers require for camera access. The datasets, video clips and extracted frames are not part of this repository.
3. Method
3.1 Data
Public data: the IndiaRDD2022 subset of RDD2022 (Arya et al., 2022), obtained through Roboflow under the CC BY 4.0 licence, with credit to the RDD2022 authors. The subset has a single class, named 0 in the source. It was treated as a pothole class after a visual check of sample images and a box-shape check (median box aspect ratio 1.59, and only 7.3 percent of boxes with an aspect ratio above 3, so crack-like boxes are rare). The split used was 3,140 training, 131 validation and 131 test images (199 labelled instances in the test split). The subset covers 1,309 base images, and the training split includes augmentation applied by Roboflow.
Local data: all video clips were filmed by the author on foot and sampled at about one frame per second. Frames were annotated by hand in Roboflow under one rule: clear holes and broken-surface patches are boxed, and crack-only frames are left empty. The labelling of some large patches on dark asphalt could not be verified against a single consistent rule, which is reported as a limitation.
Set	Content
Local train v1	9 clips filmed on 3 October 2026, 359 images (287 training, 72 validation) split at random by frame
Local train v2	18 clips, 547 training and 179 validation images. The nine new clips were filmed on 8 and 9 October 2026. Six went to training and three whole clips (107 frames) to validation. The validation set also holds the original 72 frames, which share clips with the training images, so it is only partly independent.
Local test	50 images from clips recorded on 27 and 30 September 2026, 161 boxes, held out from all training
Local preprocessing was Auto-Orient and a stretch to 640 by 640 pixels, with no augmentation.
3.2 Training
All models start from COCO-pretrained YOLOv8n weights. The public baseline was trained on the public split, and the local models were obtained by fine-tuning that baseline on the local data (seed 0, Google Colab T4). The baseline used Ultralytics 8.4.159 and later experiments used 8.4.174. Both versions are reported because exports and validation differ slightly between releases. A gentle fine-tune (backbone frozen with freeze=10, SGD, low learning rate) was tested as a way to reduce forgetting of the public data.
3.3 Compression and export
Models were exported to LiteRT through the Ultralytics exporter. The proposal named TensorFlow.js, but Ultralytics replaced its tfjs export with LiteRT in version 8.4.83, so the exported .tflite files are run in the browser with LiteRT.js instead. The INT8 variant uses static post-training quantisation of weights and activations. It was calibrated on the local training images (287 images were found, slightly below the 300 recommended by the exporter). An earlier INT8 export calibrated on only four generic images was discarded after the fault was identified.
3.4 In-browser inference
The page loads the model with LiteRT.js and runs it on the CPU through WebAssembly. Each frame is stretched to 640 by 640 pixels without padding, as in training, converted to RGB in channel-first order and scaled to the range 0 to 1. The model output has shape (1, 5, 8400); boxes above the confidence threshold are filtered with non-maximum suppression at an IoU of 0.5 in the page itself. All computation takes place on the device and no frame leaves it.
4. Results
4.1 Accuracy (mAP at IoU 0.5)
PyTorch weights, 50-image local test set and public test split:
Model	Local mAP50	Local F1	Public mAP50
Public baseline	0.035	0.098	0.290
Fine-tuned v1 (9 clips)	0.211	0.325	0.042
Gentle fine-tune (freeze=10)	0.121	0.187	0.178
Fine-tuned v2 (18 clips)	0.200	0.289	0.062
Exported LiteRT files, the configurations compared in the browser:
Config	Public mAP50	Local mAP50	Local precision	Local recall	Local F1
A, public FP32	0.269	0.043	0.082	0.131	0.101
B, public INT8	0.250	0.063	0.091	0.124	0.105
C, fine-tuned FP32	0.029	0.230	0.335	0.323	0.329
D, fine-tuned INT8	0.026	0.215	0.281	0.286	0.283
Version v1 was carried forward. A paired bootstrap over the 50 test images (F1 at confidence 0.25) gave a difference between v2 and v1 of [-0.153, 0.023], an interval that includes zero, so the two are statistically indistinguishable and the choice of v1 is a judgement and not a demonstrated improvement.
4.2 Compression trade-off
The models served in the demo are configurations C and D.
	FP32	INT8
Local mAP50	0.230	0.215
Local F1	0.329	0.283
File size	12.3 MB	3.3 MB (3.7 times smaller)
PC median latency per frame	358 ms	328 ms
Redmi 14C median latency per frame	1422 ms (0.69 fps)	1196 ms (0.83 fps)
Redmi 14C 95th percentile	1481 ms	1215 ms
Redmi 14C wall time for 500 frames	722.7 s	604.2 s
Battery drop over 500 frames (phone)	2 percentage points	2 percentage points
Energy estimate from run time (phone)	baseline	about 16 percent lower
Quantisation reduced local mAP50 by 1.5 points and local F1 by 4.6 points, in exchange for a file 3.7 times smaller and a median latency about 16 percent lower on the phone and about 8 percent lower on the PC. Both precision (0.335 to 0.281) and recall (0.323 to 0.286) fell, precision by more.
Latency is the full time per frame inside the page: preprocessing, model inference and decoding. It is therefore higher than a model-only timing of the same file.
Measurement conditions: all benchmarks used Chrome, 500 consecutive frames from the device camera pointed at a static scene, and a confidence threshold of 0.25. The phone and the laptop were both unplugged. Each final 500-frame phone benchmark was run once per model. On the laptop, the first INT8 run was repeated because other applications were slowing the machine (334 ms, then 328 ms); the repeat is reported.
The energy figure is an estimate, not a measurement. Both runs lost two percentage points of battery, and the readout has a resolution of one percent, so the two models cannot be separated by battery drain. The estimate assumes similar power draw during both runs and uses the ratio of run times (604.2 s to 722.7 s).
Two earlier 100-frame pilot runs on the same phone (also unplugged) are reported for transparency and were not used in any comparison, because they were shorter and the camera scene was not controlled.
Run	Model	Frames	Median (ms)	p95 (ms)	Min to max (ms)	Wall time (s)
Pilot	FP32	100	1575	4736	1413 to 10537	194.0
Pilot	INT8	100	1525	1562	1368 to 1581	145.2
Final	FP32	500	1422	1481	1387 to 2345	722.7
Final	INT8	500	1196	1215	1162 to 1780	604.2
In the pilot, INT8 was about 3 percent faster at the median, against 16 percent in the final runs. The on-page battery field showed 100 percent (charging) during the pilots, which conflicts with the 75 to 76 percent in the phone status bar, so no battery data from the pilots are used.
4.3 Public versus local domain gap
The public baseline reaches 0.290 mAP50 on its own test split but only 0.035 on local roads, a drop of more than eight-fold. Fine-tuning on local imagery raises local mAP50 to 0.211 and lowers public mAP50 to 0.042. For the exported FP32 files, public mAP50 falls from 0.269 to 0.029, about nine-fold, which confirms catastrophic forgetting. Evaluating the PyTorch model directly gave the same conclusion. On a further 107 frames from new roads, both v1 and v2 found only 8 percent of the labelled objects, so generalisation beyond the roads in the training clips remains weak.
The gap should not be read as a pure India versus Nigeria effect. Loose public labels, a weak baseline (0.29 mAP50 on its own test split) and differences in camera and viewpoint also contribute.
4.4 Example detections on a phone
Both images show the same pothole, photographed from the same position with a Redmi 14C, after the final benchmark runs, at a confidence threshold of 0.4.
FP32	INT8
![FP32 detection](docs/images/fp32_pothole_conf0.4.png)	![INT8 detection](docs/images/int8_pothole_conf0.4.png)
Score 0.84	Score 0.50
Both models detect the pothole at the chosen threshold, but the INT8 model reports a markedly lower confidence. A stricter threshold of about 0.6 would remove the INT8 detection and keep the FP32 one. These images illustrate behaviour on a single pothole and are not evidence of overall accuracy. In earlier close-up views, where the pothole filled the whole frame, scores were only 0.26 to 0.30, which indicates sensitivity to viewing distance.
5. Running the demo
Open the demo link in Chrome, on a phone or a computer.
Select a model (INT8 or FP32) and press Load model. The first load downloads the file, so a connection is needed once.
Use Choose File to test a single image, Start camera for live detection, or Benchmark 100 frames and Benchmark 500 frames to measure latency. The benchmark takes its frames from the camera.
Allow camera access when the browser asks. Camera access requires HTTPS or localhost.
The page was tested in Chrome. Some privacy-oriented browsers block the script that loads the runtime and may fail to load the model.
To run it locally, place demo.html and both model files in one folder and start a local server from that folder:
    python -m http.server 8000 --bind 127.0.0.1

Then open http://127.0.0.1:8000/demo.html.
6. Reproducing the experiments
Open the notebook in Google Colab. Run the setup cell first: it installs the packages, mounts Google Drive and downloads the datasets from Roboflow. Then run the sections in order. The Roboflow key must be stored in Colab Secrets under the name ROBOFLOW_API_KEY. No credential is stored in this repository.
The training cells take a long time on a T4. Because library versions and GPU runs differ, rerunning them may give slightly different numbers from those reported above.
7. Ethics and data protection
All video was recorded by the author on foot. No face or number plate appears in any image published in this repository. The local video clips, extracted frames and annotated datasets are not published; they are held in private Roboflow projects. All inference runs on the device and no image is uploaded or stored by the demo. Dataset licences were respected, and the small size of the local test set is disclosed below.
8. Limitations
The local test set contains 50 images and 161 boxes, so small differences between models are within noise. The set was inspected while several models were compared, so the reported local figures are slightly optimistic. Labelling consistency for large patches on dark asphalt could not be fully verified. The validation set is only partly independent, because 72 of its frames share clips with the training images. Doubling the number of clips (v2) did not improve accuracy on the local test set. Latency was measured on one phone model (Redmi 14C) and one laptop, with one final benchmark per configuration (500 frames) and two short pilot runs on the phone, under the stated conditions. The browser runs on the CPU through WebAssembly, and a GPU backend was not evaluated. At about 0.8 frames per second on the phone, the present build is not real time. The INT8 calibration set was slightly smaller than recommended. Weather, lighting and road-surface coverage are incomplete, and the model generalises poorly to roads absent from the training clips.
Optional components described in the proposal, namely a local IndexedDB queue, Firebase synchronisation and a Leaflet dashboard map, are not included in this release.
9. Future work
Reduce the input resolution or evaluate a GPU backend to approach real-time operation on low-cost phones. Collect more varied local data across seasons and road types, and enlarge the test set. Compare with YOLOv8s. Add the geotagged offline queue and synchronisation layer.
10. Acknowledgements and use of AI tools
The public data comes from RDD2022 (Arya et al., 2022). An AI assistant (Claude, by Anthropic) was used to help plan the experiments, write and debug the notebook and browser code, and draft this documentation. The data collection, annotation, training runs, measurements on the phone and the decisions reported here are the author's own, and every figure in this document was checked against the saved outputs.
References
Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., and Sekimoto, Y. (2022). RDD2022: A multi-national image dataset for automatic road damage detection. arXiv:2209.08538.
Jocher, G., Chaurasia, A., and Qiu, J. (2023). YOLO by Ultralytics (v8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics
Terven, J. R., Cordova-Esparza, D. M., and Romero-Gonzalez, J. A. (2023). A comprehensive review of YOLO architectures in computer vision: From YOLOv1 to YOLOv8 and YOLO-NAS. Machine Learning and Knowledge Extraction, 5(4), 1680-1716.
