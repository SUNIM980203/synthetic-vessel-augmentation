# Data and restricted-artifact access

No imagery, checkpoints, per-image predictions, embeddings, exact scene/patch membership, third-party cutouts or generated training images are included. This is a release-scope decision; it is NOT a claim that every artifact is legally prohibited from redistribution.

Dataset-specific restrictions and authorized acquisition must be checked by the authors. See `../historical_v45_1/THIRD_PARTY_DATA.md` and the archived split-construction metadata. xView, HRSC2016-MS and the DIOR public-mirror evaluation must remain distinguished. Do not substitute another DIOR partition and claim exact replication.

The excluded materials remain outside the public release; no separate access service is offered for these files. Public research-material inquiries may be sent to 20257555@gs.cwnu.ac.kr; this contact does not imply access to withheld artifacts. Hashes identify retained sources but do not reconstruct missing membership or grant access rights.

## Upstream acquisition entry points

- xView: https://www.xview.us/ ; follow the original xView detection dataset link, not xView2/xView3. Historical acquisition required registration. Current download/login availability is not guaranteed by this repository; obtain authorization from the distributor. Challenge terms: https://xviewdataset.org/terms.html . Those challenge terms are not a blanket redistribution license.
- HRSC2016-MS: https://github.com/wmchen/HRSC2016-MS ; the maintainers link Google Drive/Baidu downloads and request citation of MSSDet (2022), DOI 10.3390/rs14215460. Follow the maintainers' instructions and resolve reuse permissions with them; a download link alone is not a license.
- DIOR: https://gcheng-nwpu.github.io/#Datasets ; use the dataset authors' DIOR entry and acquisition instructions. The study's public-mirror evaluation partition is separately bounded in the historical records. Obtaining another DIOR distribution does not establish exact partition equivalence.

These landing pages were checked during public-package preparation. Source data were neither downloaded nor revalidated. Users must establish their own authorized access. Exact end-to-end replay is not offered because the retained private inputs will not be supplied.

Public seed-level aggregate statistics support recomputation of the supplied contrasts and mean intervals. Independent AP recomputation and scene bootstrap require withheld per-image/source data; full training additionally requires authorized imagery, split membership, initialization, assets and the matching environment.
