# WSC-SSWM

## Summary
This script uses Random Forest machine learning to automatically extract water bodies from RCM data. The Random Forest 
machine learning allows flexiblity for the script to be easily modified for any other Satellite input.

## Monitoring dynamic waterbodies in Canada 
The National Hydrologic Services (NHS) of Canada is responsible for the upkeep and distribution of data from over
2500 gauging stations strategically placed across the country to monitor Canadian water resources. While the data these
stations collect is important to the regions they are located in, a majority of water bodies and systems in this country
are completely unmonitored. Many of these water bodies, especially within the prairie regions, exibit dramatic shifts
in waterbody volume and extent thoughout the year, which can have significant impacts on the surrounding region.
Thus, the included algorithm was developed to monitor these dynamic waterbodies using SAR, specifically data from
the RADARSAT Constellation Mission (RCM).

## Radarsat Constellation Mission
RCM is a trio of radar satellites launched in June of 2019, that began collecting data in early 2020. SAR was chosen
for an operational monitoring program such as this due to its ability to image at any time of day and through cloud
cover, versus optical sensors that require sunlight and direct sight of the surface. Unlike Canada’s previous generations
of radar satellite (RADARSAR-1 launched in 1995, and RADARSAT-2 launched in 2007), having three satellites in a
constellation allows for access to nearly the entire country each day. This large accessability allows NHS to monitor
highly volatile waterbodies, such as those in the prairies, on a biweekly basis with high spatial resolution products
(5m).  **Other satellite source data can easily be incorporated with only small modificaitons to the code to ingest them 
to the correct format**

## Random forest machine learning 
The waterbodies themselves are detected using open-source Random Forest machine learning which, because of its
robustness, is able to accommodate a variety of SAR sensors and beam-modes. The algorithm is trained on the
Global Surface Water (GSW) Dataset (https://global-surface-water.appspot.com), which gives each 30x30m pixel on
Earth a occurrence value between 0 and 100. These values were derrived from 30 years of Landsat data, and give a
percentage of how often the pixel was water in the record. Scenes are trained on themselves using pixels within the
scene determined from GSW to have a greater than 90% probability of being water. Results are then filtered to remove
false-positives, and presented in both raster and vector formats. Additional details on the methods of the algorithm are
presented by Millard et al. 2020.

## This repository
This repository contains all the nessesary Python and Cython files to run the algorithm, which consists of two parts:
1) Preprocessing SAR scenes, mainly calibration, speckle-filtering, orthorectification, and energy-band creation. 
2) Classifying preprocessed SAR data as water/non-water using the RF model. 

### Preparing your environment

Information on how to prepare your environment to run the algorithm is presented in the included documentation PDF.

## Reference
[Koreen Millard, Nicholas Brown, Douglas Stiff & Alain Pietroniro (2020): Automated surface water detection from
space: a Canada-wide, open-source, automated, near-real time solution, Canadian Water Resources Journal / Revue
canadienne des ressources hydriques, DOI: 10.1080/07011784.2020.1816499](https://doi.org/10.1080/07011784.2020.1816499)


