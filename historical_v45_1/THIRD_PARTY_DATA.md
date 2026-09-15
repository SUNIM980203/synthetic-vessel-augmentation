# Third-party data and restricted inputs

| Dataset/input | Study role | Redistributed? | Authorized acquisition and local reconstruction |
|---|---|---|---|
| xView | Training, historical validation, and test evaluation | No | Obtain the imagery and labels from the official xView distributor under its terms. The archive documents split construction, aggregate split metadata, and the retained canonical split hash. Exact scene/patch membership is not included because the retained record does not establish redistribution permission for the source identifiers. Authorized researchers must reconstruct or localize study data from authorized source data and the documented construction metadata; the retained hash permits comparison with an independently recovered candidate mapping but does not by itself guarantee exact reconstruction. |
| HRSC2016-MS | Secondary external vessel transfer evaluation | No | Obtain HRSC2016 from its authorized distributor under the applicable terms, then reconstruct the evaluation layout described by the retained protocol. |
| DIOR public mirror | Secondary external transfer evaluation | No | Obtain DIOR through an authorized source under its terms. The manuscript explicitly bounds the mirror/partition interpretation. |
| Third-party vessel source cutouts | Historical synthetic-data construction | No | Reuse requires access to the original authorized source assets and their licensing/provenance records. The archive supplies method and aggregate audit records, not the rasters. |
| Locally retained historical frames/artifacts | Same-scale feasibility and provenance audit | No | These are not public inputs. The archive supplies frozen decisions, hashes, aggregate gate outputs, and stopping records; exact replay requires the retained local artifacts. |

No third-party copyrighted imagery is bundled. Prepared metadata never grants redistribution rights to the underlying data. Any future repository/code/documentation license applies only to materials the authors are authorized to license and does not override third-party dataset or asset licenses.
