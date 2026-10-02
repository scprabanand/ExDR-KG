## Identifier systems per node type
| node_type      |   entities | id_system(s)                                    | example_ids                                    |
|:---------------|-----------:|:------------------------------------------------|:-----------------------------------------------|
| drug           |       1605 | DrugBank (100.0%)                               | DB00201, DB09019, DB01267                      |
| gene           |      21000 | Entrez Gene (100.0%)                            | 5345, 9409, 10848                              |
| disease        |        270 | DOID (50.7%); UMLS/MedDRA C-code (49.3%)        | DOID:14227, DOID:9352, DOID:8778               |
| protein        |         92 | Symbol/name (100.0%)                            | BE0000076, BE0000065, BE0000059                |
| pathway        |       4692 | Reactome (61.2%); Symbol/name (38.8%)           | PC7_3805, PC7_3964, PC7_1532                   |
| anatomy        |        402 | UBERON (100.0%)                                 | UBERON:0001533, UBERON:0000020, UBERON:0001690 |
| side_effect    |       6061 | UMLS/MedDRA C-code (100.0%); Symbol/name (0.0%) | C0023448, C0575810, C0025193                   |
| symptom        |        438 | Symbol/name (100.0%)                            | D020150, D006606, D014826                      |
| pharm_class    |        345 | ATC/pharm class (100.0%)                        | N0000007632, N0000011272, N0000178477          |
| bio_process    |      11381 | Gene Ontology (100.0%)                          | GO:0032474, GO:0010536, GO:0010759             |
| cell_component |       1391 | Gene Ontology (100.0%)                          | GO:0000784, GO:0034677, GO:0098798             |
| mol_function   |       2884 | Gene Ontology (100.0%)                          | GO:0031753, GO:0030619, GO:0018812             |

## Conflicting (up- vs down-regulation) pairs
| relation_pair                          |   pairs_in_both |   edges_up |   edges_down |   pct_of_smaller |
|:---------------------------------------|----------------:|-----------:|-------------:|-----------------:|
| anatomy upregulates/downregulates gene |               0 |      97842 |       102239 |                0 |
| drug upregulates/downregulates gene    |               0 |      18754 |        21100 |                0 |
| disease upregulates/downregulates gene |               0 |       7730 |         7623 |                0 |

## Duplicate edges
| edge_type                              |   edges |   duplicate_edges |
|:---------------------------------------|--------:|------------------:|
| protein|encoded_by|gene                |     322 |               236 |
| gene|rev_encoded_by|protein            |     322 |               236 |
| gene|participates_reactome|pathway     |  152990 |             16277 |
| pathway|rev_participates_reactome|gene |  152990 |             16277 |

## Log lines
| log          | message                                                                       |
|:-------------|:------------------------------------------------------------------------------|
| build_kg.log | Hetionet: 47,031 nodes  4,234,558 edges  (skipped 0 unmapped)                 |
| build_kg.log | DrugBank: 343 protein nodes  343 drug-targets-protein edges added             |
| build_kg.log | DisGeNET: 193 disease-associated_with-gene edges added                        |
| build_kg.log | Reactome: 152993 gene-participates_reactome-pathway edges                     |
| build_kg.log | Reactome: 2886 pathway-is_parent_of-pathway edges                             |
| build_kg.log | SIDER: 327 new side-effect nodes registered                                   |
| build_kg.log | Drug features: (1605, 2048)  (pseudo-fingerprint)                             |
| build_kg.log | Gene features (GO-SVD): (21000, 256)                                          |
| build_kg.log | Gene features (GO-SVD): (21000, 256)                                          |
| build_kg.log | Drug-disease subgraph: 14971 nodes  87425 edges  40 components  largest=14891 |
| build_kg.log | anatomy                    402 nodes                                          |
| build_kg.log | bio_process             11,381 nodes                                          |
| build_kg.log | cell_component           1,391 nodes                                          |
| build_kg.log | disease                    270 nodes                                          |
| build_kg.log | drug                     1,605 nodes                                          |
| build_kg.log | gene                    21,000 nodes                                          |
| build_kg.log | mol_function             2,884 nodes                                          |
| build_kg.log | pathway                  4,692 nodes                                          |
| build_kg.log | pharm_class                345 nodes                                          |
| build_kg.log | protein                     92 nodes                                          |
| build_kg.log | side_effect              6,061 nodes                                          |
| build_kg.log | symptom                    438 nodes                                          |
| build_kg.log | TOTAL                   50,561 nodes                                          |
| build_kg.log | (anatomy, downregulates, gene)  edges= 102,239  density=1.21e-02              |
| build_kg.log | (anatomy, expresses, gene)  edges= 526,386  density=6.24e-02                  |
| build_kg.log | (anatomy, rev_localizes, disease)  edges=   3,598  density=3.31e-02           |
| build_kg.log | (anatomy, upregulates, gene)  edges=  97,842  density=1.16e-02                |
| build_kg.log | (bio_process, rev_participates_bp, gene)  edges= 559,489  density=2.34e-03    |
| build_kg.log | (cell_component, rev_participates_cc, gene)  edges=  73,565  density=2.52e-03 |
| build_kg.log | (disease, associated_with, gene)  edges=     193  density=3.40e-05            |
| build_kg.log | (disease, associates, gene)  edges=  12,623  density=2.23e-03                 |
| build_kg.log | (disease, downregulates, gene)  edges=   7,623  density=1.34e-03              |
| build_kg.log | (disease, localizes, anatomy)  edges=   3,598  density=3.31e-02               |
| build_kg.log | (disease, presents, symptom)  edges=   3,351  density=2.83e-02                |
| build_kg.log | (disease, resembles, disease)  edges=     543  density=7.45e-03               |
| build_kg.log | (disease, rev_palliates, drug)  edges=     389  density=8.98e-04              |
| build_kg.log | (disease, rev_resembles, disease)  edges=     543  density=7.45e-03           |
| build_kg.log | (disease, rev_treats, drug)  edges=     755  density=1.74e-03                 |
| build_kg.log | (disease, upregulates, gene)  edges=   7,730  density=1.36e-03                |
| build_kg.log | (drug, binds, gene)  edges=  11,571  density=3.43e-04                         |
| build_kg.log | (drug, causes, side_effect)  edges= 138,927  density=1.43e-02                 |
| build_kg.log | (drug, downregulates, gene)  edges=  21,100  density=6.26e-04                 |
| build_kg.log | (drug, palliates, disease)  edges=     389  density=8.98e-04                  |
| build_kg.log | (drug, resembles, drug)  edges=   6,486  density=2.52e-03                     |
| build_kg.log | (drug, rev_includes, pharm_class)  edges=   1,028  density=1.86e-03           |
| build_kg.log | (drug, rev_resembles, drug)  edges=   6,486  density=2.52e-03                 |
| build_kg.log | (drug, targets, protein)  edges=     342  density=2.32e-03                    |
| build_kg.log | (drug, treats, disease)  edges=     755  density=1.74e-03                     |
| build_kg.log | (drug, upregulates, gene)  edges=  18,754  density=5.56e-04                   |
| build_kg.log | (gene, covaries, gene)  edges=  61,690  density=1.40e-04                      |
| build_kg.log | (gene, interacts, gene)  edges= 147,164  density=3.34e-04                     |
| build_kg.log | (gene, participates_bp, bio_process)  edges= 559,489  density=2.34e-03        |
| build_kg.log | (gene, participates_cc, cell_component)  edges=  73,565  density=2.52e-03     |
| build_kg.log | (gene, participates_mf, mol_function)  edges=  97,219  density=1.61e-03       |
| build_kg.log | (gene, participates_pathway, pathway)  edges=  84,371  density=8.56e-04       |
| build_kg.log | (gene, participates_reactome, pathway)  edges= 152,990  density=1.55e-03      |
| build_kg.log | (gene, regulates, gene)  edges= 265,672  density=6.02e-04                     |
| build_kg.log | (gene, rev_associated_with, disease)  edges=     193  density=3.40e-05        |
| build_kg.log | (gene, rev_associates, disease)  edges=  12,623  density=2.23e-03             |
| build_kg.log | (gene, rev_binds, drug)  edges=  11,571  density=3.43e-04                     |
| build_kg.log | (gene, rev_covaries, gene)  edges=  61,690  density=1.40e-04                  |
| build_kg.log | (gene, rev_downregulates, anatomy)  edges= 102,239  density=1.21e-02          |
| build_kg.log | (gene, rev_downregulates, disease)  edges=   7,623  density=1.34e-03          |
| build_kg.log | (gene, rev_downregulates, drug)  edges=  21,100  density=6.26e-04             |
| build_kg.log | (gene, rev_encoded_by, protein)  edges=     322  density=1.67e-04             |
| build_kg.log | (gene, rev_expresses, anatomy)  edges= 526,386  density=6.24e-02              |
| build_kg.log | (gene, rev_interacts, gene)  edges= 147,164  density=3.34e-04                 |
| build_kg.log | (gene, rev_upregulates, anatomy)  edges=  97,842  density=1.16e-02            |
| build_kg.log | (gene, rev_upregulates, disease)  edges=   7,730  density=1.36e-03            |
| build_kg.log | (gene, rev_upregulates, drug)  edges=  18,754  density=5.56e-04               |
| build_kg.log | (mol_function, rev_participates_mf, gene)  edges=  97,219  density=1.61e-03   |
| build_kg.log | (pathway, is_parent_of, pathway)  edges=   2,886  density=1.31e-04            |
| build_kg.log | (pathway, rev_participates_pathway, gene)  edges=  84,371  density=8.56e-04   |
| build_kg.log | (pathway, rev_participates_reactome, gene)  edges= 152,990  density=1.55e-03  |
| build_kg.log | (pharm_class, includes, drug)  edges=   1,028  density=1.86e-03               |
| build_kg.log | (protein, encoded_by, gene)  edges=     322  density=1.67e-04                 |
| build_kg.log | (protein, rev_targets, drug)  edges=     342  density=2.32e-03                |
| build_kg.log | (side_effect, rev_causes, drug)  edges= 138,927  density=1.43e-02             |
| build_kg.log | (symptom, rev_presents, disease)  edges=   3,351  density=2.83e-02            |
| build_kg.log | Total edges: 4,545,138                                                        |

Counters to add to src/data/kg_builder.py (log them per source and per node type) so R-E3 can be filled completely:
  1. n_matched[type][source]      entities matched to an existing node through a shared identifier (DrugBank id, Entrez id, DOID/MONDO, Reactome id)
  2. n_unmapped_dropped[type][source]   source records whose identifier could not be mapped and were dropped (write the ids to a TSV)
  3. n_duplicate_edges_merged     identical (head, relation, tail) contributed by two or more sources, merged into one edge (write source pairs)
  4. n_conflicts                  (head, tail) pairs with opposite relations, plus the rule applied (keep both / keep higher-confidence / drop) and how many kept or removed
  5. the identifier mapping tables actually used (file names and row counts)
