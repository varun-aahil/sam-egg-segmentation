# Dataset Re-tagging Audit (`tag_changes.md`)

This audit documents all image categorization corrections performed following systematic visual review of category contact sheets (`report/figures/category_sheets/*.jpg`).
Heuristic color/brightness-based classification previously misclassified non-hand objects (such as orange container lids) as hands, and green patterned cloth as herbs.

**Total Images Re-tagged:** 62 / 110

| Image Stem | Split | Previous Category | Corrected Category | Visual Verification Reason |
|:---|:---:|:---|:---|:---|
| `not_damaged_1` | `dev` | Tabletop / high-contrast background | **Hand holding egg / fingers occluding** | Egg held in open human hand; previously misclassified under tabletop. |
| `not_damaged_2` | `dev` | Tabletop / high-contrast background | **Hand holding egg / fingers occluding** | Egg held between fingers; previously misclassified under tabletop. |
| `not_damaged_4` | `dev` | White egg on white tray / fridge interior | **Tabletop / high-contrast background** | Egg resting on yellow/blue plaid cloth tablecloth; previously misclassified under white tray. |
| `not_damaged_5` | `dev` | White egg on white tray / fridge interior | **Tabletop / high-contrast background** | Egg resting on yellow/blue plaid cloth tablecloth; previously misclassified under white tray. |
| `not_damaged_6` | `dev` | White egg on white tray / fridge interior | **Tabletop / high-contrast background** | Egg resting on yellow/blue plaid cloth tablecloth; previously misclassified under white tray. |
| `not_damaged_15` | `eval` | Backlit / extreme shadows | **Tabletop / high-contrast background** | Egg resting on purple patterned cloth tablecloth; previously misclassified under backlit. |
| `not_damaged_16` | `dev` | Hand holding egg / fingers occluding | **Tabletop / high-contrast background** | Egg resting against wired earphones on patterned tablecloth; no hand present. |
| `not_damaged_17` | `dev` | Backlit / extreme shadows | **Tabletop / high-contrast background** | Egg resting on black laptop keyboard; previously misclassified under backlit. |
| `not_damaged_19` | `dev` | Backlit / extreme shadows | **Tabletop / high-contrast background** | Egg resting on black laptop keyboard; previously misclassified under backlit. |
| `not_damaged_21` | `dev` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on pink plastic bowl / container; retagged to container / lid / plain surface. |
| `not_damaged_22` | `dev` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on pink plastic container with CHERRY label; retagged to container / lid / plain surface. |
| `not_damaged_25` | `dev` | Tabletop / high-contrast background | **Hand holding egg / fingers occluding** | Egg held in palm of human hand; previously misclassified under tabletop. |
| `not_damaged_29` | `dev` | Hand holding egg / fingers occluding | **Tabletop / high-contrast background** | Egg resting on marble floor with strong cast shadow; no hand present. |
| `not_damaged_30` | `dev` | Backlit / extreme shadows | **Egg on container / lid / plain surface** | Egg resting on plain white cardboard shipping package; previously misclassified under backlit. |
| `not_damaged_31` | `dev` | White egg on white tray / fridge interior | **Egg on container / lid / plain surface** | Egg resting on bright red plastic food container lid; previously misclassified under white tray. |
| `not_damaged_32` | `eval` | Backlit / extreme shadows | **Egg on container / lid / plain surface** | Egg resting on plain white cardboard package; previously misclassified under backlit. |
| `not_damaged_34` | `eval` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on pink plastic bowl / container; retagged to container / lid / plain surface. |
| `not_damaged_35` | `dev` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on pink plastic bowl / container; retagged to container / lid / plain surface. |
| `not_damaged_39` | `eval` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on pink plastic container with CHERRY label; retagged to container / lid / plain surface. |
| `not_damaged_40` | `dev` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on pink plastic bowl with white paper liner; retagged to container / lid / plain surface. |
| `not_damaged_43` | `eval` | Backlit / extreme shadows | **Tabletop / high-contrast background** | Egg resting on black laptop keyboard; previously misclassified under backlit. |
| `not_damaged_52` | `dev` | Hand holding egg / fingers occluding | **Tabletop / high-contrast background** | Egg resting against wired earphones on patterned cloth; no hand present. |
| `not_damaged_62` | `dev` | Hand holding egg / fingers occluding | **Tabletop / high-contrast background** | Egg resting against wired earphones on patterned cloth; no hand present. |
| `not_damaged_95` | `dev` | Hand holding egg / fingers occluding | **Tabletop / high-contrast background** | Egg resting on textured purple patterned fabric; no hand present. |
| `not_damaged_100` | `dev` | Hand holding egg / fingers occluding | **Tabletop / high-contrast background** | Egg resting on plain polished marble tile floor; no hand present. |
| `not_damaged_101` | `dev` | Hand holding egg / fingers occluding | **Tabletop / high-contrast background** | Egg resting on plain polished marble tile floor; no hand present. |
| `not_damaged_107` | `dev` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on white styrofoam bead fill in dish; retagged to container / plain surface. |
| `not_damaged_108` | `eval` | Hand holding egg / fingers occluding | **Egg on container / lid / plain surface** | Egg resting on wall ledge adjacent to orange container lid; no hand present. |
| `not_damaged_109` | `dev` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on white printed cardboard carton; retagged to container / plain surface. |
| `not_damaged_110` | `dev` | Hand holding egg / fingers occluding | **Egg on container / lid / plain surface** | Egg resting on rim of orange food container / bucket; no hand present. |
| `not_damaged_113` | `dev` | Multiple touching eggs | **Egg in boiling water / submerged** | Two eggs submerged in water inside metal cooking pot; retagged to boiling water / submerged. |
| `not_damaged_114` | `eval` | Multiple touching eggs | **Egg in boiling water / submerged** | Two eggs submerged in water inside metal cooking pot; retagged to boiling water / submerged. |
| `not_damaged_116` | `eval` | Tabletop / high-contrast background | **White egg on white tray / fridge interior** | Egg in molded white plastic refrigerator egg tray; previously misclassified under tabletop. |
| `not_damaged_117` | `eval` | Tabletop / high-contrast background | **White egg on white tray / fridge interior** | Egg in molded white plastic refrigerator egg tray; previously misclassified under tabletop. |
| `not_damaged_118` | `dev` | Egg in boiling water / submerged | **Egg on container / lid / plain surface** | Egg resting on shiny foil food pouch packaging; previously misclassified under boiling water. |
| `not_damaged_119` | `eval` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on shiny foil food pouch packaging; retagged to container / plain surface. |
| `not_damaged_120` | `dev` | Hand holding egg / fingers occluding | **Egg on container / lid / plain surface** | Egg resting on orange food-container lid placed on blue bucket; no hand present. |
| `not_damaged_122` | `eval` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on white styrofoam bead fill in dish; retagged to container / plain surface. |
| `not_damaged_126` | `eval` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on white styrofoam bead fill in dish; retagged to container / plain surface. |
| `not_damaged_127` | `eval` | Hand holding egg / fingers occluding | **Egg on container / lid / plain surface** | Egg resting on wall ledge adjacent to orange container lid; no hand present. |
| `not_damaged_128` | `eval` | Multiple touching eggs | **Egg in boiling water / submerged** | Two eggs submerged in water inside metal cooking pot; retagged to boiling water / submerged. |
| `not_damaged_129` | `dev` | Multiple touching eggs | **Egg in boiling water / submerged** | Two eggs submerged in water inside metal cooking pot; retagged to boiling water / submerged. |
| `not_damaged_130` | `eval` | Hand holding egg / fingers occluding | **Egg on container / lid / plain surface** | Egg sitting inside orange food container lid over blue bucket; no hand present. |
| `not_damaged_131` | `dev` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on shiny foil food pouch packaging; retagged to container / plain surface. |
| `not_damaged_138` | `dev` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on shiny foil food pouch packaging; retagged to container / plain surface. |
| `not_damaged_139` | `dev` | Egg in boiling water / submerged | **Egg on container / lid / plain surface** | Egg resting on container rim / lid beside wall; previously misclassified under boiling water. |
| `not_damaged_141` | `dev` | Hand holding egg / fingers occluding | **Egg on container / lid / plain surface** | Egg resting on orange food container lid over blue bucket; no hand present. |
| `not_damaged_143` | `dev` | Egg in boiling water / submerged | **White egg on white tray / fridge interior** | Egg resting on white printed paper bag; previously misclassified under boiling water. |
| `not_damaged_149` | `eval` | Egg in boiling water / submerged | **Egg on container / lid / plain surface** | Egg resting on inverted metal pot lid beside tiles; previously misclassified under boiling water. |
| `not_damaged_150` | `dev` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on rim of metal pot / bowl; retagged to container / plain surface. |
| `not_damaged_153` | `dev` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on white styrofoam bead fill in dish; retagged to container / plain surface. |
| `not_damaged_154` | `dev` | Foliage / herbs in background (specular highlights) | **Tabletop / high-contrast background** | Egg resting on green patterned cloth tablecloth; no herbs or foliage present. |
| `not_damaged_155` | `dev` | White egg on white tray / fridge interior | **Hand holding egg / fingers occluding** | Egg held in hand above background; previously misclassified under white tray. |
| `not_damaged_156` | `dev` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on blue plastic container / lid; retagged to container / plain surface. |
| `not_damaged_157` | `dev` | Tabletop / high-contrast background | **Hand holding egg / fingers occluding** | Egg held in fingers of human hand; previously misclassified under tabletop. |
| `not_damaged_158` | `dev` | White egg on white tray / fridge interior | **Hand holding egg / fingers occluding** | Egg held in fingers of human hand; previously misclassified under white tray. |
| `not_damaged_159` | `dev` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting in shallow stainless steel dish; retagged to container / plain surface. |
| `not_damaged_160` | `dev` | Tabletop / high-contrast background | **Hand holding egg / fingers occluding** | Egg held in fingers of human hand; previously misclassified under tabletop. |
| `not_damaged_161` | `dev` | Tabletop / high-contrast background | **Hand holding egg / fingers occluding** | Egg held in open palm of human hand; previously misclassified under tabletop. |
| `not_damaged_162` | `dev` | Foliage / herbs in background (specular highlights) | **Tabletop / high-contrast background** | Egg resting on green patterned cloth tablecloth; no herbs or foliage present. |
| `not_damaged_163` | `dev` | Tabletop / high-contrast background | **Egg on container / lid / plain surface** | Egg resting on blue plastic container / lid; retagged to container / plain surface. |
| `not_damaged_164` | `eval` | Foliage / herbs in background (specular highlights) | **Tabletop / high-contrast background** | Egg resting on green patterned cloth tablecloth; no herbs or foliage present. |