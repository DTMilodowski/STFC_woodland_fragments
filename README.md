STFC_woodland_fragments
=======================
Summary
------
This repository contains code produced during the STFC project, undertaken as part of the Scottish Earth Observation Service (SEOS). The aim of this project was to explore the extent to which we can map woodland fragments in Scottish landscapes using Sentinel 1 and Sentinel 2 data. The basic workflow was to (1) produce a reference map and characterise tree cover with 1-metre resolution gridded airborne laser scanning data; (2) fit a random forest classifier that maps tree cover using Sentinel data; (3) undertake an accuracy assessment with spatially independent cross-validation. Preprocessing of the data layers was carried out prior to this analysis.

Code
------
- src/accuracy_assessment/accuracy_assessment.py <i>Contains scripts to carry out various accuracy assessment procedures</i>
- src/data_io/data_io.py <i>Contains scripts to help write xarray objects to geotiff rasters</i> 
- src/lidar_reference_classification.py <i>Creation of the lidar-based reference classification</i>
- src/woodland_classification/single_tree_woodland_classification.py <i>Contains the routine to fit and cross validate random forest classifier exploiting Sentinel data, including accuracy assessments</i>
- src/woodland_classification/woodland_classification_figures.py <i>Contains the code for producing the figures from the summary report</i>
