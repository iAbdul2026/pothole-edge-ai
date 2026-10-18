Smartphone-Based Edge AI for Real-Time Pothole and Road-Defect Detection on Nigerian Roads
A measured comparison of FP32 and INT8 in-browser inference on a low-cost phone
Case Study 6, COEN872: AI Systems Development, Department of Computer Engineering, Ahmadu Bello University, Zaria (M.Sc. 2025/26)
Author: Abdulhamid Musa (Reg. No. P25EGCP8002)
Source code and notebook: https://github.com/iAbdul2026/pothole-edge-ai
Live demo (open in Chrome): https://iabdul2026.github.io/pothole-edge-ai/demo.html
Contents
Abstract
Introduction and Justification
Value Proposition and Expected Impact
Technical Solution
Demonstration and Results
Scalability Plan
Product and Service Marketability
Conclusion, Limitations and Future Work
Running the demo
Reproducing the experiments
Repository contents
Use of AI tools
References
Abstract
Pothole and road-surface damage degrade vehicle safety and impose recurrent maintenance costs, yet in Nigeria defects are still reported through ad hoc complaints and sporadic inspection drives. This study investigates whether a compressed object detector can run entirely inside a smartphone web browser, without installation or a server and, once loaded, without a live connection, and quantifies the accuracy, latency, energy and model-size trade-offs involved.
A YOLOv8n detector, pre-trained on COCO and then on an India subset of RDD2022, was fine-tuned on locally filmed road video from Zaria and Kano, exported as a LiteRT FP32 model and a statically quantised INT8 model, and deployed with LiteRT.js on a low-cost Redmi 14C. On a held-out set of 50 local images, fine-tuning raised mean average precision at an IoU of 0.5 (mAP50) from 0.035 to 0.211, while mAP50 on the public test split fell from 0.290 to 0.042, an instance of catastrophic forgetting. INT8 quantisation reduced the model file by a factor of 3.7 and the median per-frame time by about 16 percent on the phone, at a cost of 1.5 mAP50 points. The phone sustains between 0.7 and 0.8 frames per second, so the prototype is a proof of concept and not a real-time system. Because the local test set is small (161 boxes), all conclusions are qualified accordingly.
Keywords: edge AI; object detection; YOLOv8; post-training quantisation; domain shift; catastrophic forgetting; pothole detection; Nigeria
Key findings at a glance
Research question	Finding	How firm it is
RQ1 Compression trade-off	INT8 is 3.7 times smaller (12.3 to 3.3 MB) and about 16% faster per frame on the Redmi 14C (1196 against 1422 ms). It costs 1.5 mAP50 points and 4.6 F1 points.	Size is exact. Speed rests on one 500-frame run per model. Energy is estimated from run time, not measured.
RQ2 Domain gap	The public baseline falls from 0.290 mAP50 on its own test split to 0.035 on local roads (more than eight-fold). Local fine-tuning lifts it to 0.211.	Moderate. 50 images, 161 boxes. The gap is mixed with label quality and camera differences.
RQ3 Retention	Fine-tuning cuts public mAP50 from 0.290 to 0.042. A frozen backbone keeps 0.178 but gains only 0.121 locally. Doubling the local clips gave no gain.	Single runs, no seed variance. The v1 and v2 models cannot be told apart statistically.
Readiness	About 0.7 to 0.8 frames per second and a local F1 of about 0.33: a working proof of concept whose output needs human review.	Measured directly, on one phone and one laptop.
1. Introduction and Justification
1.1 Problem statement
Pothole formation and surface failure on Nigerian roads outpace repair. The consequences include vehicle damage, congestion and elevated crash risk, and the reporting chain is weak: complaints reach agencies through social media, residents' letters and occasional inspection drives, none of which yields a consistent, geo-referenced and prioritised defect inventory. Automating the perception step with a learned detector that runs on commodity smartphones could convert routine journeys into distributed inspections and give agencies an evidence base for allocating scarce maintenance funds.
1.2 The Nigerian context
Road network and maintenance. The Federal Roads Maintenance Agency (FERMA) reported in 2023 that funding was a challenge in maintaining about 36,000 km of federal highways (Gazette Nigeria, 2023). In February 2025 the Minister of State for Works stated that more than 35,000 km of the federal network cannot be financed through annual budgets alone (AllAfrica, 2025b). No recent, network-wide, published road-condition survey was located.
Road safety. The Federal Road Safety Corps (FRSC) recorded 10,446 road traffic crashes and 5,289 deaths in 2025 (Ariyo, 2026). The World Health Organization estimated 36,722 road deaths in Nigeria in 2021 (World Health Organization, 2023b). No national statistic isolating pothole-related crashes was found, so the road-safety argument rests on network condition and the maintenance gap.
Connectivity. There were 195.11 million active mobile subscriptions in July 2026, but 4G and 5G together accounted for only 58.92 percent of connections in June 2026 (Techeconomy, 2026). A design that needs a live connection for every frame is least dependable where it is most needed.
1.3 Why an AI-based, on-device approach
Recognising a pothole is a visual pattern-recognition problem with large intra-class variability: depth, outline, wetness, lighting and surrounding surface all vary, and shadows, patches and worn asphalt are frequent confounders. No compact hand-written rule separates these cases, which is why convolutional detectors have become the leading approach (Ma et al., 2022; Safyari et al., 2024). Executing the model on the device removes the dependence on connectivity during inference and the per-request cost of a cloud service, keeps images on the phone, and, when delivered as a web page, removes installation as a barrier to participation.
1.4 Gaps addressed
First, the public benchmark RDD2022 does not include Nigerian roads, so the size of the domain gap there is unmeasured. Second, reported mobile results seldom pair accuracy with in-browser latency and energy for FP32 and INT8 variants of the same detector on a low-cost phone. Third, the interaction between local fine-tuning and retention of benchmark accuracy is rarely reported for road damage. The full related-work discussion is in the report.
1.5 Aim and research questions
The aim is to build and evaluate a smartphone-based, in-browser pothole detector suited to low-connectivity conditions. The word real-time in the title follows the wording of the case study; at the measured speed of about 0.7 to 0.8 frames per second (Section 4.6) the prototype does not reach it.
RQ1 (trade-off). What are the trade-offs in accuracy, latency, energy and model size between a 32-bit floating-point (FP32) and an 8-bit quantised (INT8) version of a lightweight detector running on a low-cost phone?
RQ2 (domain gap). How large is the accuracy gap between a public benchmark subset and locally filmed Nigerian road imagery, and how much of it does local fine-tuning close?
RQ3 (retention). What does local fine-tuning cost in accuracy on the public data, and does a gentler fine-tuning regime mitigate that cost?
2. Value Proposition and Expected Impact
Beneficiaries. Motorists and transport operators bear the costs of vehicle damage and delay. Road agencies, including state ministries of works, local government road authorities, FERMA and the FRSC, need defect evidence to prioritise limited maintenance budgets. Citizens currently have no low-friction channel for reporting defects with a location and an image.
Unique value. Inference runs on the phone, so detection works without a live connection once the page and model have loaded. Delivery as a web page removes installation. And the detector was evaluated on locally filmed Nigerian roads, so the study quantifies how far an off-the-shelf model falls short in this setting instead of assuming that it will transfer.
Expected impact. Inexpensive evidence on where defects are densest could help agencies rank interventions. This is a hypothesis and not a result: the project demonstrates the detection step only. The current accuracy (an F1 score of about 0.33 on local roads) implies that the output should support human review and not replace inspection.
Improvement over existing alternatives. Manual complaint channels are slow and unstructured. Cloud-based vision systems depend on constant connectivity and typically incur per-request or licensing costs. The approach taken here removes the connectivity dependence and the per-request cost, at the price of a smaller model with lower accuracy and a lower frame rate.
3. Technical Solution
3.1 System architecture
Data acquisition and labelling. A public benchmark subset (the India portion of RDD2022, obtained through Roboflow) and locally filmed road video, sampled into frames and labelled by hand in Roboflow.
Training. Ultralytics YOLOv8n on a Google Colab T4 GPU: COCO-pretrained weights, then a public baseline, then fine-tuning on local frames.
Export and compression. The PyTorch weights are exported to LiteRT as an FP32 model and as an INT8 model calibrated on road images.
Delivery. A static web page and the two model files are hosted on GitHub Pages, which provides the HTTPS origin that mobile browsers require for camera access.
On-device inference. In Chrome, LiteRT.js executes the model on the CPU through WebAssembly. Each camera frame is stretched to 640 by 640 pixels, converted to RGB values in [0, 1], passed through the network, filtered by a confidence threshold and non-maximum suppression, and drawn as boxes on a canvas.
Planned but not implemented. An offline IndexedDB queue of geotagged detections, synchronisation to a Firebase backend, and a Leaflet map for agencies.
3.2 The AI component
The solution is AI-driven because the perception step is a learned model. The web page contains no hand-written rule for what a pothole looks like: it prepares pixels, thresholds the model's scores and removes overlapping boxes. The model is YOLOv8n, the smallest YOLOv8 variant, a single-stage detector with a convolutional backbone, a feature-pyramid neck that fuses three scales, and an anchor-free, decoupled detection head (Jocher et al., 2023; Terven et al., 2023). The fused network has 3,005,843 parameters and 8.1 GFLOPs at 640 by 640 pixels. For a single class, the output tensor has shape (1, 5, 8400): four box values and one pothole score for each of the 8,400 candidates. Training uses transfer learning from COCO-pretrained weights, first to the public road data and then to local data.
3.3 Data, labelling and preprocessing
Public data. The India subset of RDD2022 (Arya et al., 2024a), obtained through Roboflow under the CC BY 4.0 licence, with credit to the RDD2022 authors. It has 1,309 base images and a single class named 0. The class was treated as a pothole class after a visual check of 12 random training images and a box-shape check (median box aspect ratio 1.59; only 7.3 percent of boxes above 3, so crack-like boxes are rare). The split used was 3,140 training, 131 validation and 131 test images (199 labelled instances in the test split); the training count includes augmentation applied by Roboflow.
Local data. All local video was filmed by the author on foot in Zaria and Kano, so no recording was made while driving, and frames were sampled at about one frame per second. Frames were labelled by hand in Roboflow under one rule: clear holes and broken-surface patches are boxed, and frames showing only cracks are left empty. Local preprocessing was Auto-Orient and a stretch to 640 by 640 pixels with no augmentation. The labelling of some large patches on dark asphalt could not be verified against one consistent rule, which is reported as a limitation.
Table 1. Local datasets used in this study.
Set	Content
Local train v1	9 clips filmed on 3 October 2026, 359 images (287 training, 72 validation), split at random by frame
Local train v2	18 clips, 547 training and 179 validation images. Nine further clips were filmed on 8 and 9 October 2026: six went to training and three whole clips (107 frames) to validation. The validation set also holds the original 72 frames, which share clips with the training images, so it is only partly independent.
Local test	50 images (161 boxes) from clips filmed on 27 and 30 September 2026, held out from all training and used only for final results
3.4 Training and evaluation protocol
All models start from COCO-pretrained YOLOv8n weights. The public baseline was trained on the public split for up to 50 epochs at 640 pixels with batch size 16, early-stopping patience 15 and seed 0 (Google Colab, T4 GPU). Local models were obtained by fine-tuning that baseline with the same settings. Three variants were run: fine-tuning on the 9-clip data (v1); a gentler fine-tune with the backbone frozen (freeze=10), SGD and a learning rate of 0.001 for 30 epochs, intended to reduce forgetting; and fine-tuning on the 18-clip data (v2). The baseline used Ultralytics 8.4.159 and later runs used 8.4.174, and both are reported because exports and validation differ slightly between releases. Each configuration was run once, so seed variance is not estimated.
3.5 Compression and export
Models were exported to LiteRT through the Ultralytics exporter. The proposal named TensorFlow.js, but Ultralytics replaced its tfjs export with LiteRT in version 8.4.83, so the exported .tflite files are run in the browser with LiteRT.js (Yu et al., 2026). The INT8 variant uses static post-training quantisation of both weights and activations (Gholami et al., 2022), calibrated on 287 local training images, slightly below the 300 recommended by the exporter. A first INT8 export, calibrated by mistake on only four generic images, was discarded once the fault was found.
3.6 In-browser inference
The page loads the model with LiteRT.js and runs it on the CPU through WebAssembly. The input is a float32 tensor of shape (1, 3, 640, 640). Candidates above the confidence threshold are filtered with non-maximum suppression at an IoU of 0.5 in JavaScript. The page offers three modes: a single-image check, live camera detection, and a benchmark that times 100 or 500 consecutive frames and records the battery level before and after. A WebGPU backend was attempted and failed because the test page did not initialise a WebGPU device, so all reported results use the CPU.
3.7 Innovation
The contribution is not a new architecture. It is a measured, reproducible comparison of FP32 and INT8 variants of one detector running in a phone browser on a low-cost device, tied to a quantified public-versus-local domain gap and to an explicit measurement of forgetting, with negative results reported openly: doubling the number of local clips did not improve test accuracy, and the models generalise poorly to roads absent from the training clips.
3.8 Ethics and potential bias
Dataset bias. The public data come from India, with different cameras, road types and labelling conventions, and the local data come from a few days of filming on a limited set of roads in Zaria and Kano in dry conditions. The model is likely to be biased toward the roads and viewpoints it has seen, and the results show it.
Privacy. All video was recorded by the author on foot. Local frames and clips are not published; they are held in private Roboflow projects. The original proposal commits to blurring faces and number plates before storage, and a review of the local frames for identifiable faces and plates was still pending at the time of writing. All inference runs on the device and no image is uploaded by the demo.
Licensing and attribution. The public data are used under the CC BY 4.0 licence with credit to the RDD2022 authors.
4. Demonstration and Results
4.1 The prototype
The prototype is a single web page, `docs/demo.html`, served over HTTPS from GitHub Pages together with two model files, `best.tflite` (FP32, about 12 MB) and `best_int8.tflite` (about 3.3 MB). The user selects a model, loads it, and then either selects a photograph, starts the live camera, or runs a benchmark of 100 or 500 camera frames. Detections are drawn as boxes with their scores, and the page reports the time spent in preprocessing, model execution and decoding, together with the battery level before and after a benchmark. It was tested in Chrome on a Windows laptop and on a Redmi 14C, a low-cost Android phone. The page is the detection component of the road-condition mapper named in the case study; the geotagged queue, synchronisation and map are not implemented (Section 3.1).
4.2 Evaluation protocol
Accuracy was measured on two held-out sets: the public test split (131 images, 199 instances) and the local test set of 50 images (161 boxes), whose clips share no frames with any training data. The metrics are precision, recall, F1 score and mAP50, where a detection is a true positive if its IoU with an unmatched ground-truth box is at least 0.5. Because the task has a single class, mAP50 equals the average precision at IoU 0.5. To judge whether differences between models are real, a paired bootstrap over the 50 test images was used (2,000 resamples, F1 at a fixed confidence of 0.25). Latency was measured inside the page as the full time per frame, covering preprocessing, inference and decoding, over 500 consecutive camera frames pointed at a static scene, with a confidence threshold of 0.25. Chrome was used throughout, and the phone and the laptop were both unplugged. The local test set was kept untouched during training but was inspected while several models were compared, so the local figures are slightly optimistic.
4.3 Accuracy
Table 2. Accuracy of the PyTorch weights on the local and public test sets.
Model	Local mAP50	Local F1	Public mAP50
Public baseline	0.035	0.098	0.290
Fine-tuned v1 (9 clips)	0.211	0.325	0.042
Gentle fine-tune (freeze=10)	0.121	0.187	0.178
Fine-tuned v2 (18 clips)	0.200	0.289	0.062
Table 3. Accuracy of the exported LiteRT configurations A to D, the configurations compared in the browser.
Config	Public mAP50	Local mAP50	Local precision	Local recall	Local F1
A, public FP32	0.269	0.043	0.082	0.131	0.101
B, public INT8	0.250	0.063	0.091	0.124	0.105
C, fine-tuned FP32	0.029	0.230	0.335	0.323	0.329
D, fine-tuned INT8	0.026	0.215	0.281	0.286	0.283
On local roads, the best model retrieves roughly one third of the labelled potholes (recall 0.323), and roughly one third of its detections are correct (precision 0.335). This modest operating point is the basis for the position that its output should be subject to human review.
4.4 Domain gap and forgetting (RQ2 and RQ3)
The public baseline reaches 0.290 mAP50 on its own test split but only 0.035 on local roads, a drop of more than eight-fold. Fine-tuning on local imagery lifts local mAP50 to 0.211 and lowers public mAP50 to 0.042 (0.269 to 0.029 for the exported FP32 files, about nine-fold), which is catastrophic forgetting. The gentler fine-tune with a frozen backbone retained more public accuracy (0.178) but gained less locally (0.121), so it moved along the stability-plasticity trade-off and did not remove it. The gap should not be read as a pure India-versus-Nigeria effect: loose public labels, a weak baseline and differences in camera and viewpoint also contribute, and these factors could not be separated in this study.
4.5 Choosing between the 9-clip and 18-clip models
On the 50 local test images the 18-clip model (v2) did not improve on v1 (mAP50 0.200 against 0.211; F1 0.289 against 0.325). The bootstrap difference in F1 between v2 and v1 was [-0.153, 0.023]. The interval includes zero, so the two models are statistically indistinguishable on this test set. On 107 validation frames from the new roads, which neither model had seen, both found only 8 percent of the labelled objects. Version v1 was carried forward as the simpler option with exports already evaluated. This is a judgement and not a demonstrated improvement.
4.6 Compression trade-off (RQ1)
The models served in the demo are configurations C and D. The frame rates in Table 4 are 500 frames divided by the wall time of the run. The FP32 file in Table 4 scores 0.230 local mAP50 against 0.211 for the PyTorch weights of the same model in Table 2; the reason was not investigated, and Section 3.4 notes that validation differs slightly between Ultralytics releases.
Table 4. FP32 versus INT8 (configurations C and D): accuracy, size, latency and energy.
	FP32	INT8

Local mAP50	0.230	0.215
Local F1	0.329	0.283
File size	12.3 MB	3.3 MB (3.7 times smaller)
Laptop median time per frame	358 ms	328 ms
Redmi 14C median time per frame	1422 ms (0.69 fps)	1196 ms (0.83 fps)
Redmi 14C 95th percentile	1481 ms	1215 ms
Redmi 14C wall time for 500 frames	722.7 s	604.2 s
Battery drop over 500 frames (phone)	2 percentage points	2 percentage points
Energy estimate from run time (phone)	baseline	about 16 percent lower
Quantisation cost 1.5 points of local mAP50 and 4.6 points of local F1, in exchange for a file 3.7 times smaller and a median time per frame about 16 percent shorter on the phone and about 8 percent shorter on the laptop. Both precision (0.335 to 0.281) and recall (0.323 to 0.286) fell, precision by more. INT8 kept 93 percent of the FP32 local mAP50 and 86 percent of its F1 at 27 percent of the file size. For this setting the size saving is arguably the more consequential benefit, because the 9.0 MB difference matters most on slow connections. The speed gain is real but modest: at about 1.2 s per frame neither model comes near real time.
The energy figure is an estimate and not a measurement: both runs lost two percentage points of battery and the readout has a resolution of one percentage point, so the two models cannot be separated by battery drain. The estimate assumes similar power draw in both runs and uses the ratio of run times. Each final 500-frame phone benchmark was run once per model, so no uncertainty can be given for the phone timings. On the laptop, the first INT8 run was repeated because other applications were slowing the machine (334 ms, then 328 ms); the repeat is reported.
Table 5. Redmi 14C benchmark runs: earlier 100-frame pilots (reported for transparency, not used in any comparison) and the final 500-frame runs.
Run	Model	Frames	Median (ms)	p95 (ms)	Min to max (ms)	Wall time (s)
Pilot	FP32	100	1575	4736	1413 to 10537	194.0
Pilot	INT8	100	1525	1562	1368 to 1581	145.2
Final	FP32	500	1422	1481	1387 to 2345	722.7
Final	INT8	500	1196	1215	1162 to 1780	604.2
During the pilots the on-page battery field showed 100 percent (charging), which conflicts with the 75 to 76 percent in the phone status bar, so no battery data from the pilots are used.
4.7 Example detections on the phone
Both images show the same pothole, photographed from the same position with a Redmi 14C after the final benchmark runs, at a confidence threshold of 0.4.
FP32	INT8
![FP32 detection](docs/images/fp32_pothole_conf0.4.png)	![INT8 detection](docs/images/int8_pothole_conf0.4.png)
Score 0.84	Score 0.50
Both models detect the pothole at the chosen threshold, but the INT8 model reports a markedly lower confidence. A stricter threshold of about 0.6 would remove the INT8 detection and keep the FP32 one, so the operating threshold should be chosen per model. These images illustrate behaviour on a single pothole and are not evidence of overall accuracy. In earlier close-up views, where the pothole filled the whole frame, scores were only 0.26 to 0.30, which indicates sensitivity to viewing distance.
4.8 Threats to validity
Internal validity is limited by the small local test set (50 images from five clips, 161 boxes), by its repeated inspection during model comparison, by a validation set that is only partly independent, and by single runs without seed variance. External validity is limited by one phone model, one laptop, dry-season filming on a small set of roads in Zaria and Kano, and a labelling rule that was hard to apply consistently to large patches. Construct validity is limited for energy, which is inferred from run time and not measured. At about 0.7 to 0.8 frames per second on a low-cost phone, the present build is not real time.
5. Scalability Plan
Technical scalability. Inference executes on the user's phone, so adding users adds no inference load on any server. The page and models are static assets; the free GitHub Pages hosting used here suits a pilot, and a host backed by a content delivery network would suit wider use. The same code runs on any phone whose browser supports WebAssembly and camera access. One qualification applies: the current page fetches the LiteRT.js runtime from a CDN at load time and does not cache the page, runtime and models for offline use, so a connection is needed to start it. Inference itself needs none.
Infrastructure required for deployment. A pilot needs HTTPS static hosting for the page and models; phones carried by volunteers or agency staff; a small backend that receives geotagged detections when a connection is available (the planned Firebase layer); and an agency dashboard that maps them (the planned Leaflet map). The last two do not yet exist.
Table 6. Challenges to scaling, supporting evidence from this study, and mitigations.
Challenge	Evidence from this study	Mitigation
Phone speed	1.2 to 1.4 s per frame on a Redmi 14C	Smaller input size, frame skipping, and GPU or NPU backends (Yu et al., 2026), each followed by a new accuracy measurement
Road and seasonal variation	Version v2 gave no gain, and both local models found 8 percent of objects on new roads	Periodic local recollection, a held-out test set per area, and continual-learning methods to limit forgetting (Wang et al., 2024)
Label consistency	Large patches on dark asphalt were hard to label by one rule	A written labelling guide and a second labeller on a sample of frames
Privacy at scale	Frames can contain people and number plates	On-device inference, blurring before any image is shared, and federated learning as an option for training across agencies (Dwivedi et al., 2024)
Dependence on a connection at load time	The runtime is fetched from a CDN	Self-host the runtime and cache page and models with a service worker
Expansion within Nigeria and beyond. The approach extends to other states, local government areas or countries by collecting local imagery and retraining, because the page and delivery mechanism are unchanged. Additional defect classes such as cracking, rutting or edge break would need labelled examples of each, and the present model detects a single pothole class. The results show that moving to a new region without local data is not safe: the public baseline lost more than eight-fold in accuracy on local roads.
6. Product and Service Marketability
Deployment model. A realistic route is a partnership or licence with an agency that already owns the maintenance problem: a state ministry of works, a local government road authority, FERMA or the FRSC. The service would pair a citizen-facing reporting layer, which is the phone detector, with an agency-facing dashboard that ranks road sections by detected defects. The dashboard is planned and not yet built, so this describes a model and not a product ready for sale.
Funding and revenue. Possible revenue streams are per-agency dashboard subscriptions and grant funding tied to road-safety programmes. The rationale for demand exists at several levels: the FRSC recorded 5,289 deaths in 2025 (Ariyo, 2026); the World Health Organization reports 1.19 million road deaths a year worldwide (World Health Organization, 2023a); and federal officials state that more than 35,000 km of roads cannot be financed from annual budgets alone (AllAfrica, 2025b). No funder or agency was approached and no price was tested, so these are arguments for demand and not evidence of it.
Competitive landscape. The alternatives are commercial inspection systems that depend on constant connectivity and are typically costly, and the manual complaint channels described in Section 1. Research prototypes for smartphone pothole detection exist (Amri & Kusuma, 2025), but a detailed survey of commercial competitors was not carried out. The proposed differentiator is a zero-install, on-device approach that tolerates poor connectivity during inference and has been evaluated on local roads. Its main weaknesses today are modest accuracy and a low frame rate on a budget phone.
Go-to-market. The plan starts with a single-LGA pilot using locally collected data, followed by outreach to neighbouring authorities. The pilot's first objective would be to measure accuracy against an agency's own survey of the same roads. The Zuba-Kaduna-Zaria corridor, with its recorded crash burden (Leadership, 2026), is a natural candidate for a first evaluation.
Costs. Development cost so far consisted of the author's time, a free Colab GPU and free hosting. Pilot costs would consist mainly of data collection, labelling time and dashboard hosting, none of which was measured. In a crowdsourced model users bring their own phones, so device cost is not borne by the service. As an indication of device capability, the phone used in this study is a low-cost model.
7. Conclusion, Limitations and Future Work
7.1 Conclusion
In answer to RQ1, INT8 quantisation reduced the model from 12.3 MB to 3.3 MB and the median per-frame time by about 16 percent on a Redmi 14C, at a cost of 1.5 mAP50 points and 4.6 F1 points; energy was estimated from run time (about 16 percent lower) but not measured. In answer to RQ2, the public baseline lost more than eight-fold in mAP50 on local roads, and fine-tuning raised local mAP50 from 0.035 to 0.211. In answer to RQ3, that gain came at the cost of public accuracy (0.290 to 0.042), a gentler regime retained more (0.178) but gained less locally (0.121), and doubling the number of local clips gave no measurable improvement. The prototype shows that in-browser detection is feasible on a low-cost phone, at about 0.7 to 0.8 frames per second and with modest accuracy.
On this evidence, INT8 is the more sensible default for the browser demo, chiefly because of the 9.0 MB smaller download on slow connections, while FP32 remains the choice where each detection should score as high as possible. Because quantisation shifts the score distribution, the confidence threshold should be set separately for each model. This recommendation is a judgement drawn from one phone and one test set, not a tested deployment result.
7.2 Limitations
The local test set has 50 images and 161 boxes from five clips, so neighbouring frames are correlated and small differences between models are within noise, and it was inspected while several models were compared. Labelling consistency for large patches on dark asphalt could not be fully verified. The validation set is only partly independent, because 72 of its frames share clips with the training images. Latency was measured on one phone model and one laptop, with one final benchmark per configuration (500 frames) and two short pilot runs on the phone. The browser runs on the CPU, a GPU backend was not evaluated, and offline operation after the first load was not implemented or tested. The INT8 calibration set was slightly smaller than recommended. Weather, lighting and road-surface coverage are incomplete, and the local clips come from Zaria and Kano only. Compression was limited to post-training INT8 quantisation of one detector, YOLOv8n. The offline queue, synchronisation layer and map described in the proposal were not built. Finally, no national statistic linking potholes to crashes was found, so the road-safety motivation is argued from the condition of the network.
Table 7. Differences between the proposal and the delivered work.
Item	Proposal (16 September 2026)	Delivered
Browser format	Export to TensorFlow.js	LiteRT (.tflite) run with LiteRT.js, because Ultralytics replaced its tfjs export in version 8.4.83
Models compared	YOLOv8n FP32 against INT8, optionally YOLOv8s	YOLOv8n FP32 against INT8; YOLOv8s was not trained
Offline queue, synchronisation and map	IndexedDB queue, Firebase backend, Leaflet dashboard map	Not implemented
Local training data	About 300 to 500 images from about 10 to 15 clips in Zaria and Kano	359 images from 9 clips for the model carried forward; a second model on 18 clips was tested and not adopted
Local test data	30 to 50 images, held out	50 images, held out from all training
Filming	Stationary phone or passenger, never the driver	Filmed by the author on foot; no recording while driving
Energy	Energy proxy, or battery drain over a number of inferences	Battery drain over 500 frames (2 percentage points for both models) and an estimate from run time
Blurring of faces and plates	Before storage	Review of local frames still pending (Section 3.8)
7.3 Future work
The first priority is to reduce the per-frame time, through a smaller input size, frame skipping, and the GPU and NPU backends of LiteRT.js (Yu et al., 2026), and to measure the accuracy cost of each. The second is to collect more varied local data across seasons and road types, to enlarge the test set, and to compare with YOLOv8s. The third is to add self-hosted runtime caching so that the page works offline, followed by the geotagged queue, synchronisation and dashboard map. Continual-learning methods (Wang et al., 2024) could reduce forgetting when new local data are added. Finally, the local frames should be reviewed for faces and number plates, and any found should be blurred before images are shared.
Running the demo
Open the demo link in Chrome, on a phone or a computer.
Select a model (INT8 or FP32) and press Load model. The first load downloads the file, so a connection is needed once.
Use Choose File to test a single image, Start camera for live detection, or Benchmark 100 frames and Benchmark 500 frames to measure latency. The benchmark takes its frames from the camera.
Allow camera access when the browser asks. Camera access requires HTTPS or localhost.
The page was tested in Chrome. Some privacy-oriented browsers block the script that loads the runtime and may fail to load the model.
To run it locally, place `demo.html` and both model files in one folder and start a local server from that folder:
```
python -m http.server 8000 --bind 127.0.0.1
```
Then open http://127.0.0.1:8000/demo.html.
Reproducing the experiments
Open the notebook in Google Colab. Run the setup cell first: it installs the packages, mounts Google Drive and downloads the datasets from Roboflow. Then run the sections in order. The Roboflow key must be stored in Colab Secrets under the name `ROBOFLOW_API_KEY`. No credential is stored in this repository. Training used a T4 GPU, seed 0, Ultralytics 8.4.159 for the baseline and 8.4.174 for later runs. The training cells take a long time on a T4, and because library versions and GPU runs differ, rerunning them may give slightly different numbers from those reported above. The local datasets are held in private Roboflow projects and are not published.
Repository contents
Path	Description
`docs/demo.html`	Browser application: image check, live camera detection and benchmark mode
`docs/best.tflite`	Fine-tuned YOLOv8n (v1), FP32 export (about 12 MB)
`docs/best_int8.tflite`	Fine-tuned YOLOv8n (v1), INT8 export (about 3.3 MB)
`docs/images/`	Example detections used in this README
`COEN872_pothole_yolov8_colab.ipynb`	Training, evaluation and export notebook (Google Colab, T4 GPU)
The docs folder is served through GitHub Pages, which provides the HTTPS origin that mobile browsers require for camera access. The datasets, video clips and extracted frames are not part of this repository.
Use of AI tools
An AI assistant (Claude, by Anthropic) was used to help plan the experiments, write and debug the notebook and browser code, and draft this documentation. The data collection, annotation, training runs, measurements on the phone and the decisions reported here are the author's own, and every figure in this document was checked against the saved outputs.
References
Africa Check. (2026, February 4). Factsheet: Road safety in Nigeria in wake of high-profile crash. AllAfrica. https://allafrica.com/stories/202602040536.html
AllAfrica. (2025a, October 21). Nigeria: FRSC records reported 6,858 road crashes in 9 months. https://allafrica.com/stories/202510210121.html
AllAfrica. (2025b, February 19). Nigeria: Govt seeks private investment in Nigeria's 35,000km federal road infrastructure. https://allafrica.com/stories/202502190093.html
Amri, A. U., & Kusuma, G. P. (2025). Comparative study of pothole detection using deep learning on smartphone. Indonesian Journal of Electrical Engineering and Computer Science, 37(2), 995-1004. https://doi.org/10.11591/ijeecs.v37.i2.pp995-1004
Ariyo, I. (2026). FRSC records 10,446 crashes, 5,289 deaths in 2025. News Agency of Nigeria. https://nannews.ng/frsc-records-10446-crashes-5289-deaths-in-2025/
Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Omata, H., Kashiyama, T., & Sekimoto, Y. (2022). Crowdsensing-based road damage detection challenge (CRDDC'2022). In Proceedings of the 2022 IEEE International Conference on Big Data (Big Data). https://arxiv.org/abs/2211.11362
Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., & Sekimoto, Y. (2024a). RDD2022: A multi-national image dataset for automatic road damage detection. Geoscience Data Journal, 11(4), 846–862. https://doi.org/10.1002/gdj3.260
Arya, D., Omata, H., Maeda, H., & Sekimoto, Y. (2024b). ORDDC'2024: State of the art solutions for optimized road damage detection. In Proceedings of the 2024 IEEE International Conference on Big Data (BigData). https://doi.org/10.1109/BigData62323.2024.10825254
Bello-Salau, H., Onumanyi, A. J., Adebiyi, R. F., Adekale, A. D., Bello-Salahuddeen, R., & Ajayi, O.-O. (2023). A critical appraisal of various implementation approaches for real-time pothole anomaly detection: Toward safer roads in developing nations. Engineering Proceedings, 56(1), Article 301. https://doi.org/10.3390/ASEC2023-15519
Dwivedi, S. K., Arya, D., & Sekimoto, Y. (2024). Road damage detection across borders: Federated learning insights from Japan, China, Norway and the USA. In Proceedings of the 2024 IEEE Smart World Congress (SWC).
Gazette Nigeria. (2023, July 9). FERMA to repair 36,000km federal roads: Official. https://gazettengr.com/ferma-to-repair-36000km-federal-roads-official/
Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M. W., & Keutzer, K. (2022). A survey of quantization methods for efficient neural network inference. In G. K. Thiruvathukal, Y.-H. Lu, J. Kim, Y. Chen, & B. Chen (Eds.), Low-power computer vision: Improve the efficiency of artificial intelligence. Chapman and Hall/CRC. https://doi.org/10.1201/9781003162810
Jocher, G., Chaurasia, A., & Qiu, J. (2023). YOLO by Ultralytics (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics
Kothai, R., Prabakaran, N., Murthy, Y. V. S., Cenkeramaddi, L. R., & Kakani, V. (2024). Pavement distress detection, classification, and analysis using machine learning algorithms: A survey. IEEE Access, 12, 126943-126960. https://doi.org/10.1109/ACCESS.2024.3455093
Leadership. (2026). 597 died in road crashes in 1 month: FRSC. https://leadership.ng/597-died-in-road-crashes-in-1-month-frsc/
Ma, N., Fan, J., Wang, W., Wu, J., Jiang, Y., Xie, L., & Fan, R. (2022). Computer vision for road imaging and pothole detection: A state-of-the-art review of systems and algorithms. Transportation Safety and Environment, 4(4), Article tdac026. https://doi.org/10.1093/tse/tdac026
Mehajabin, N., Ma, Z., Wang, Y., Tohidypour, H. R., & Nasiopoulos, P. (2022). Real-time deep learning based road deterioration detection for smart cities. In Proceedings of the 2022 18th International Conference on Wireless and Mobile Computing, Networking and Communications (WiMob) (pp. 321–326). IEEE.
Oluka, P. (2026, August 19). Nigeria internet subscriptions rise to 157.4 million as broadband penetration hits 56.1% in May. Techeconomy. https://techeconomy.ng/nigeria-internet-subscriptions-rise-to-1574-million-as-broadband-penetration-hits-561-in-may
Rey, L., Bernardos, A. M., Dobrzycki, A. D., Carramiñana, D., Bergesio, L., Besada, J. A., & Casar, J. R. (2025). A performance analysis of You Only Look Once models for deployment on constrained computational edge devices in drone applications. Electronics, 14(3), Article 638. https://doi.org/10.3390/electronics14030638
Safyari, Y., Mahdianpari, M., & Shiri, H. (2024). A review of vision-based pothole detection methods using computer vision and machine learning. Sensors, 24(17), Article 5652. https://doi.org/10.3390/s24175652
Techeconomy. (2026, September 10). Nigeria's active mobile subscriptions rise to 195.11m as data drives telecom growth. https://techeconomy.ng/mtn-crosses-100m-subscribers-as-nigerias-mobile-lines-hit-19511-million
Terven, J. R., Cordova-Esparza, D. M., & Romero-Gonzalez, J. A. (2023). A comprehensive review of YOLO architectures in computer vision: From YOLOv1 to YOLOv8 and YOLO-NAS. Machine Learning and Knowledge Extraction, 5(4), 1680-1716.
Wang, L., Zhang, X., Su, H., & Zhu, J. (2024). A comprehensive survey of continual learning: Theory, method and application (arXiv:2302.00487v3). arXiv. https://arxiv.org/abs/2302.00487
World Health Organization. (2023a). Global status report on road safety 2023. https://www.who.int/publications/i/item/9789240086517
World Health Organization. (2023b). Global status report on road safety 2023: Country and territory profiles. Nigeria. https://cdn.who.int/media/docs/default-source/country-profiles/road-safety/road-safety-2023-nga.pdf
Yu, A., Gao, Y., Xiong, Y., Liu, W., & She, J. (2025). Mobile-YOLO: A lightweight YOLO for road crack detection on mobile devices. Journal of Advanced Computational Intelligence and Intelligent Informatics, 29(6), 1443-1453. https://doi.org/10.20965/jaciii.2025.p1443
Yu, P., Ristić, M., Soulanille, M., & Parikh, C. (2026, July 9). LiteRT.js, Google's high performance Web AI inference. Google Developers Blog. https://developers.googleblog.com/litertjs-googles-high-performance-web-ai-inference/
