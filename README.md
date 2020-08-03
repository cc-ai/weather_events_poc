# weather_events_poc
Our little internal hackathon!
The model checkpoints is too large to upload. So I paste the address here:

## Heat Models path:
Masker: /network/tmp1/ccai/checkpoints/tianyu/omnigan/output_dirs/skyMaskerBest/  
[Eval Experiment](https://www.comet.ml/tianyu-z/omnigan/4a3aa6e00ee146528de88342c8e12db3?experiment-tab=images&groupBy=false&orderBy=desc&sortBy=step)  
[Masker Generation for the Painter Experiment] (https://www.comet.ml/tianyu-z/omnigan/a309a26b00b4489884b79a37caf7af85?experiment-tab=images&groupBy=false&orderBy=desc&sortBy=step)  
Painter: /network/tmp1/ccai/checkpoints/tianyu/omnigan/output_dirs/skyPainterBest/

## Related Dataset Path:
### Dataset 1 (Pure Sky)
Pure Sky without Sun: /network/tmp1/ccai/data/SunPairs/skyNs_Ojpg_allAsS1/  
Pure Sky with Sun: /network/tmp1/ccai/data/SunPairs/skyS1_all/  
Mask/Label for SPADE (Pair with **skyS1_all**): /network/tmp1/ccai/data/SunPairs/skyS1_allSpadeMask/  
Pix2Pix Standard Dataset (A is **Pure Sky without Sun**; B is **Pure Sky with Sun**): /network/tmp1/ccai/data/SunPairs/pix2pixdata_OandS1/  

### Dataset 2 (Google Street Images (filtered))
note: comes from /network/tmp1/ccai/data/munit_dataset/non_flooded/streetview_mvp/, but I filtered some of them out.  
Google Street Image (Sun in Sky): /network/tmp1/ccai/data/streetview_mvp_filterSun/SuninSky/  
Sky Mask of Google Street Image generated by Masker: /network/tmp1/ccai/data/streetview_mvp_filterSun/SkyMask/  

### Dataset 3 (Label Box 2020 and its sky mask)

### Dataset 4 (Omnigan Base with sky mask which is generated by deeplab v2)

## Model Intro
### Masker: Omnigan Masker with ADVENT and MINENT  
Data used: Dataset 4  
### Painter: Omnigan Painter
Data used: Dataset 1, 2 and 3
