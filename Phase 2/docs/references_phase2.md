# Phase 2 references (for the final report)

IEEE style. Phase 1 / Review references ([1]–[21]) are in `References/` and the Review 2 deck; this list covers what Phase 2 added. Renumber when merging (plan step 13.4).

## A. Methods

- [P1] C. Elkan and K. Noto, "Learning classifiers from only positive and unlabeled data," in *Proc. 14th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, 2008, pp. 213–220. *(Step 7, PU Elkan–Noto)*
- [P2] F. Mordelet and J.-P. Vert, "A bagging SVM to learn from positive and unlabeled examples," *Pattern Recognition Letters*, vol. 37, pp. 201–209, 2014. *(Step 7, PU bagging)*
- [P3] G. Ke *et al.*, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017. *(Steps 6–11)*
- [P4] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD*, 2016, pp. 785–794. *(Step 8.2, comparison model)*
- [P5] D. R. Roberts *et al.*, "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," *Ecography*, vol. 40, no. 8, pp. 913–929, 2017. *(Step 6, spatial / temporal hold-out)*
- [P6] M. Saerens, P. Latinne, and C. Decaestecker, "Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure," *Neural Computation*, vol. 14, no. 1, pp. 21–41, 2002. *(Step 8.4, prior correction)*
- [P7] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in *Proc. 22nd Int. Conf. Machine Learning (ICML)*, 2005, pp. 625–632. *(Step 8.4, Platt / isotonic calibration)*
- [P8] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017. *(Step 11, SHAP)*
- [P9] S. M. Lundberg *et al.*, "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, pp. 56–67, 2020. *(Step 11, TreeSHAP)*
- [P10] P. Quinn, K. Beven, P. Chevallier, and O. Planchon, "The prediction of hillslope flow paths for distributed hydrological modelling using digital terrain models," *Hydrological Processes*, vol. 5, no. 1, pp. 59–79, 1991. *(Steps 4, 9: FD8 routing, UICA)*
- [P11] R. Barnes, C. Lehman, and D. Mulla, "Priority-flood: An optimal depression-filling and watershed-labeling algorithm for digital elevation models," *Computers & Geosciences*, vol. 62, pp. 117–127, 2014. *(Step 4)*
- [P12] A. D. Nobre *et al.*, "Height Above the Nearest Drainage – a hydrologically relevant new terrain model," *Journal of Hydrology*, vol. 404, no. 1–2, pp. 13–29, 2011. *(Step 5, HAND)*
- [P13] USDA Natural Resources Conservation Service, "Urban Hydrology for Small Watersheds," Technical Release 55 (TR-55), 2nd ed., 1986. *(Steps 9.0c, 9: SCS curve number)*
- [P14] S. M. Iacus, G. King, and G. Porro, "Causal inference without balance checking: Coarsened exact matching," *Political Analysis*, vol. 20, no. 1, pp. 1–24, 2012. *(Step 10, matching)*
- [P15] J. D. Angrist and J.-S. Pischke, *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton, NJ, USA: Princeton Univ. Press, 2009. *(Step 10, difference-in-differences)*

## B. Data

- [P16] C. F. Brown *et al.*, "Dynamic World, Near real-time global 10 m land use land cover mapping," *Scientific Data*, vol. 9, art. 251, 2022. *(Step 2)*
- [P17] M. Pesaresi and P. Politis, "GHS-BUILT-S R2023A – GHS built-up surface grid, derived from Sentinel-2 composite and Landsat, multitemporal (1975–2030)," European Commission, Joint Research Centre, 2023. *(Steps 2, 9: 2030 projection)*
- [P18] C. Funk *et al.*, "The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes," *Scientific Data*, vol. 2, art. 150066, 2015. *(Steps 3, 5: CHIRPS)*
- [P19] B. Lehner and G. Grill, "Global river hydrography and network routing: baseline data and new approaches to study the world's large river systems," *Hydrological Processes*, vol. 27, no. 15, pp. 2171–2186, 2013. *(Step 4: HydroBASINS / HydroRIVERS)*
- [P20] L. Hawker *et al.*, "A 30 m global map of elevation with forests and buildings removed," *Environmental Research Letters*, vol. 17, no. 2, art. 024016, 2022. *(FABDEM)*
- [P21] National Remote Sensing Centre (NRSC), ISRO, "National Database for Emergency Management (NDEM): flood inundation layers, Kerala," [Online]. Available: https://ndem.nrsc.gov.in *(Steps 3, 6, 9.0a, 10)*
- [P22] European Space Agency, "Copernicus Sentinel-1 SAR GRD," Copernicus Programme. *(Step 3)*
- [P23] OpenStreetMap contributors, "OpenStreetMap" and "Nominatim geocoder," [Online]. Available: https://www.openstreetmap.org, https://nominatim.openstreetmap.org (accessed Nov. 26, 2026). *(Step 10b, geocoding of waterlogging spots)*
- [P24] N. Gorelick *et al.*, "Google Earth Engine: Planetary-scale geospatial analysis for everyone," *Remote Sensing of Environment*, vol. 202, pp. 18–27, 2017. *(data access)*

## C. Waterlogging reports used in Step 10b (all accessed Nov. 26, 2026)

- [P25] "Heavy rain batters Kerala, 6 killed," *Deccan Herald*. [Online]. Available: https://www.deccanherald.com/india/heavy-rain-batters-kerala-6-killed-681554.html *(MG Road, Kochi railway station)*
- [P26] "Rain fury: Cloudburst triggers heavy downpour in Kochi," *Onmanorama*, May 28, 2024. [Online]. Available: https://www.onmanorama.com/news/kerala/2024/05/28/kochi-rain-flooded-roads-hits-normal-life-cloudburst-suspected.html *(Ernakulam South station, KSRTC stand, Palarivattom, Marottichuvadu, Thammanam, Kundannoor)*
- [P27] "Flooding due to cloudburst disrupts life in Kochi, citizens taken by surprise," *The News Minute*, May 2024. [Online]. Available: https://www.thenewsminute.com/kerala/flooding-due-to-cloudburst-disrupts-life-in-kochi-citizens-taken-by-surprise *(Kaloor, Edappally, Kalamassery, Tripunithura)*
- [P28] "Kochi faces early flooding amid summer rains," *Urban Acres*, 2024. [Online]. Available: https://urbanacres.in/kochi-faces-early-flooding-amid-summer-rains/ *(Thrikkakara, Infopark Kakkanad)*
- [P29] "Heavy rains leave parts of Kochi inundated, normal life disrupted," *The News Minute*, Oct. 2019. [Online]. Available: https://www.thenewsminute.com/article/heavy-rains-leave-parts-kochi-inundated-normal-life-disrupted-110896 *(Panampilly Nagar, 21 Oct 2019)*
- [P30] "Beware Kerala, 20cm rain in 4 hours at Kochi reveals global warming's bare fangs," *Onmanorama*, Oct. 22, 2019. [Online]. Available: https://www.onmanorama.com/news/kerala/2019/10/22/beware-kerala-global-warming-bare-its-fangs.html *(rainfall of the 2019 event)*
- [P31] "Form taskforce to end waterlogging in Kochi: Kerala HC," *Onmanorama*, Oct. 23, 2019. [Online]. Available: https://www.onmanorama.com/news/kerala/2019/10/23/form-taskforce-to-end-waterlogging-in-kochi--kerala-hc.html *(official response, context)*
- [P32] "2019 Indian floods," *Wikipedia*. [Online]. Available: https://en.wikipedia.org/wiki/2019_Indian_floods *(Cochin airport closure)*

**Note.** [P25]–[P32] are news and secondary sources, used only as observed locations, not as scientific evidence. The report should cite them as data sources for Step 10b.
