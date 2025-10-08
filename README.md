# DTransKT: A Dual Transferable Knowledge Tracing Framework for Cross-Disciplinary Self-Adaptation

Knowledge Tracing (KT) is pivotal in intelligent tutoring systems, as it models the dynamic evolution of students' knowledge states based on their learning interactions. However, the cross-disciplinary generalization of existing KT models is subjected to a dual constraint: the heterogeneity in students' cognitive abilities and the divergent domain-specific knowledge structures. To address this challenge, we propose DTransKT, a dual transferable knowledge tracing framework tailored for cross-disciplinary adaptability. DTransKT enhances existing knowledge tracing models by dynamically aligning student representations and integrating external knowledge semantics. Specifically, the framework incorporates a Cross-disciplinary Graph-matching (CG) module, which captures meta-skill representations based on students' learning trajectories. Through cross-disciplinary node matching, the CG module aligns student-specific features, thereby improving tracing accuracy. Additionally, the Cross-disciplinary Attention-assisting (CA) module leverages pre-trained language models to extract meta-knowledge semantics from textual content, enhancing transferability. Comprehensive experimental evaluations demonstrate that DTransKT consistently enhances the performance of seven prominent KT models under direct transfer settings, achieving average improvements of 14.2% in accuracy (ACC) and 4.5% in area under the curve (AUC) across diverse datasets. These findings affirm the efficacy of our approach in enabling cross-disciplinary knowledge tracing transfer.

## References
### Projects

1. https://github.com/hcnoh/knowledge-tracing-collection-pytorch 
2. https://github.com/arshadshk/SAKT-pytorch 
3. https://github.com/shalini1194/SAKT 
4. https://github.com/arshadshk/SAINT-pytorch 
5. https://github.com/Shivanandmn/SAINT_plus-Knowledge-Tracing- 
6. https://github.com/arghosh/AKT 
7. https://github.com/JSLBen/Knowledge-Query-Network-for-Knowledge-Tracing 
8. https://github.com/xiaopengguo/ATKT 
9. https://github.com/jhljx/GKT 
10. https://github.com/THUwangcy/HawkesKT
11. https://github.com/ApexEDM/iekt
12. https://github.com/Badstu/CAKT_othermodels/blob/0c28d870c0d5cf52cc2da79225e372be47b5ea83/SKVMN/model.py
13. https://github.com/bigdata-ustc/EduKTM
14. https://github.com/shalini1194/RKT
15. https://github.com/shshen-closer/DIMKT
16. https://github.com/skewondr/FoLiBi
17. https://github.com/yxonic/DTransformer
18. https://github.com/lilstrawberry/ReKT

### Papers

1. DKT: Deep knowledge tracing 
2. DKT+: Addressing two problems in deep knowledge tracing via prediction-consistent regularization 
3. DKT-Forget: Augmenting knowledge tracing by considering forgetting behavior 
4. KQN: Knowledge query network for knowledge tracing: How knowledge interacts with skills 
5. DKVMN: Dynamic key-value memory networks for knowledge tracing 
6. ATKT: Enhancing Knowledge Tracing via Adversarial Training 
7. GKT: Graph-based knowledge tracing: modeling student proficiency using graph neural network 
8. SAKT: A self-attentive model for knowledge tracing 
9. SAINT: Towards an appropriate query, key, and value computation for knowledge tracing 
10. AKT: Context-aware attentive knowledge tracing 
11. HawkesKT: Temporal Cross-Effects in Knowledge Tracing
12. IEKT: Tracing Knowledge State with Individual Cognition and Acquisition Estimation
13. SKVMN: Knowledge Tracing with Sequential Key-Value Memory Networks
14. LPKT: Learning Process-consistent Knowledge Tracing
15. QIKT: Improving Interpretability of Deep Sequential Knowledge Tracing Models with Question-centric Cognitive Representations
16. RKT: Relation-aware Self-attention for Knowledge Tracing
17. DIMKT: Assessing Student's Dynamic Knowledge State by Exploring the Question Difficulty Effect
18. ATDKT: Enhancing Deep Knowledge Tracing with Auxiliary Tasks
19. simpleKT: A Simple but Tough-to-beat Baseline for Knowledge Tracing
20. SparseKT: Towards Robust Knowledge Tracing Models via K-sparse Attention
21. FoLiBiKT: Forgetting-aware Linear Bias for Attentive Knowledge Tracing
22. DTransformer: Tracing Knowledge Instead of Patterns: Stable Knowledge Tracing with Diagnostic Transformer
23. stableKT: Enhancing Length Generalization for Attention Based Knowledge Tracing Models with Linear Biases
24. extraKT: Extending Context Window of Attention Based Knowledge Tracing Models via Length Extrapolation
25. csKT: Addressing Cold-start Problem in Knowledge Tracing via Kernel Bias and Cone Attention
26. LefoKT: Rethinking and Improving Student Learning and Forgetting Processes for Attention Based Knowledge Tracing Models
27. FlucKT: Cognitive Fluctuations Enhanced Attention Network for Knowledge Tracing
28. UKT: Uncertainty-aware Knowledge Tracing
29. HCGKT: Hierarchical Contrastive Graph Knowledge Tracing with Multi-level Feature Learning
30. RobustKT: Enhancing Knowledge Tracing through Decoupling Cognitive Pattern from Error-Prone Data
