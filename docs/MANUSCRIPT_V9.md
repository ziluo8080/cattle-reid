# Manuscript v9 / 论文 v9 同步说明

Updated: 2026-09-17. [中文 Word 工作稿](manuscript/manuscript-v9.zh-CN.docx).

## 修改范围

- 以公式与图件核验稿为底稿，合入参考文献核对稿的修订：OpenCowID零样本术语、ImageNet背景与本文实现配置的区分、bootstrap原理与本文重抽样设定的区分。
- 修正参考文献第9、14条作者姓名，补齐第32条页码；保留第25条核对稿页码。下方文献表从合并后的Word提取，与提供的参考文献核对稿32条逐条一致；本次同步未再次在线检索文献。
- 正文与图表展示名称改为受限SupCon-out，简称SupCon-out；原归档标识B/B-selected不变。它是已有SupCon-out的查询锚点、支持对比集合限制实例，不是新增损失。
- 图1保留同一真实任务的候选库示例；图2注明真实照片与非实测示意的边界；图6改为青灰配色，全部12个点估计保留，列最高值50.10%、58.35%、61.82%加粗描框。
- 未更改实验结果、原生公式、候选任务、训练或评分代码。没有新训练或新模型推理。

## 文件与复现边界

- `figures/`：v9实际使用的六组SVG/PDF，图1、2、6包含真实数据照片；文字和几何元素为矢量，照片本身仍是位图，不宣称整图为纯矢量。
- `scripts/plot_paper_figures.py`：便携的数值核对绘图入口，输出到`derived/figures/`，名称与图6配色、最高值标记已同步。该脚本不读取照片，不是最终排版图的逐像素复现。请勿用简化图误覆盖最终归档图。
- `tables/`、`results/`、`protocols/`与已发布的数据和权重附件保持原始标识与实验内容不变。命令行继续使用`--method B`。
- v9仍为中文工作稿。作者、资助、伦理适用依据和利益冲突等未确认事项保留待确认状态；SideView仍为探索性外部分析，不宣称采集事件独立性已验证。
- 本次Git提交更新工作稿与展示材料，不移动既有Release标签、不替换历史实验附件、不改变仓库可见性。

## English Summary

This update merges the citation-reviewed draft and the formula/evidence audit, archives the Chinese v9 working manuscript, and synchronizes the final figures. Restricted SupCon-out is the display name for the unchanged archived B/B-selected method. The teal-gray Figure 6 emphasizes column maxima, not statistical significance. Original experimental data and command identifiers remain unchanged. The portable plotting script provides simplified numerical verification plots, not pixel-identical reconstruction of the final photo-containing artwork. This draft is not a published article or an independent confirmatory external evaluation.

## 参考文献 / References

1. Bhole, A.; Udmale, S.S.; Falzon, O.; Azzopardi, G. CORF3D contour maps with application to Holstein cattle recognition from RGB and thermal images. Expert Syst. Appl. 2022, 192, 116354. https://doi.org/10.1016/j.eswa.2021.116354

2. Andrew, W.; Gao, J.; Mullan, S.; Campbell, N.; Dowsey, A.W.; Burghardt, T. Visual identification of individual Holstein-Friesian cattle via deep metric learning. Comput. Electron. Agric. 2021, 185, 106133. https://doi.org/10.1016/j.compag.2021.106133

3. Vinyals, O.; Blundell, C.; Lillicrap, T.; Kavukcuoglu, K.; Wierstra, D. Matching Networks for One Shot Learning. In Advances in Neural Information Processing Systems; 2016, 29. Available online: https://papers.nips.cc/paper_files/paper/2016/hash/90e1357833654983612fb05e3ec9148c-Abstract.html (accessed on 16 September 2026).

4. Snell, J.; Swersky, K.; Zemel, R. Prototypical Networks for Few-shot Learning. In Advances in Neural Information Processing Systems; 2017, 30. Available online: https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html (accessed on 16 September 2026).

5. Khosla, P.; Teterwak, P.; Wang, C.; Sarna, A.; Tian, Y.; Isola, P.; Maschinot, A.; Liu, C.; Krishnan, D. Supervised Contrastive Learning. In Advances in Neural Information Processing Systems; 2020, 33, 18661–18673. Available online: https://papers.nips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html (accessed on 16 September 2026).

6. Musgrave, K.; Belongie, S.; Lim, S.N. A Metric Learning Reality Check. In Lecture Notes in Computer Science; 2020, 681–699. https://doi.org/10.1007/978-3-030-58595-2_41

7. Beery, S.; Van Horn, G.; Perona, P. Recognition in Terra Incognita. In Proceedings of the European Conference on Computer Vision; 2018, 456–473. Available online: https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Beery_Recognition_in_Terra_ECCV_2018_paper.php (accessed on 16 September 2026).

8. Möller, S. SideViewCows2026 – Dairy Cow Re-Identification Dataset. [数据集]. Zenodo, 2026. https://doi.org/10.5281/zenodo.21605650

9. Bhole, A.; Udmale, S.S.; Falzon, O.; Azzopardi, G. Recognition of Holstein Cattle with Thermal and RGB images. [数据集]. DataverseNL, 2021. https://doi.org/10.34894/7m108f

10. Čermák, V.; Picek, L.; Adam, L.; Papafitsoros, K. WildlifeDatasets: An Open-Source Toolkit for Animal Re-Identification. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision; 2024, 5953–5963. Available online: https://openaccess.thecvf.com/content/WACV2024/html/Cermak_WildlifeDatasets_An_Open-Source_Toolkit_for_Animal_Re-Identification_WACV_2024_paper.html (accessed on 16 September 2026).

11. Perneel, M.; Adriaens, I.; Verwaeren, J.; Aernouts, B. Dynamic Multi-Behaviour, Orientation-Invariant Re-Identification of Holstein-Friesian Cattle. Sensors 2025, 25, 2971. https://doi.org/10.3390/s25102971

12. Hooker, J.M.; de Medeiros, B.B.; Saha, C.; Abdulrahman, T.; Alves, A.A.C. Multitask contrastive learning for individual dairy cow recognition across different behavior classes based on small image sets. J. Dairy Sci. 2026, 109, 1800–1815. https://doi.org/10.3168/jds.2025-26731

13. Prabhune, O.; Kim, Y. OpenCowID: Zero-Shot Visual Identification of Dairy Cows. In 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV); 2026, 1491–1500. https://doi.org/10.1109/wacv61042.2026.00151

14. Zhao, J.-M.; Lian, Q.-S.; Xiong, N.N. Multi-Center Agent Loss for Visual Identification of Chinese Simmental in the Wild. Animals 2022, 12, 459. https://doi.org/10.3390/ani12040459

15. Li, G.; Erickson, G.E.; Xiong, Y. Individual Beef Cattle Identification Using Muzzle Images and Deep Learning Techniques. Animals 2022, 12, 1453. https://doi.org/10.3390/ani12111453

16. Meng, Y.; Yoon, S.; Han, S.; Fuentes, A.; Park, J.; Jeong, Y.; Park, D.S. Improving Known–Unknown Cattle’s Face Recognition for Smart Livestock Farm Management. Animals 2023, 13, 3588. https://doi.org/10.3390/ani13223588

17. Qi, H.; Song, T.; Zhao, Y. Dynamic_Bottleneck Module Fusing Dynamic Convolution and Sparse Spatial Attention for Individual Cow Identification. Animals 2025, 15, 2519. https://doi.org/10.3390/ani15172519

18. Liu, J.; Fuentes, A.; Han, S.; Yoon, S.; Jeong, Y.; Park, D.S. Learning Compact Identity Representations for Weakly Textured Hanwoo Cattle Re-Identification. Animals 2026, 16, 2320. https://doi.org/10.3390/ani16152320

19. Zhao, A.; Wu, H.; Fan, D.; Li, K. Individual Cow Recognition Based on Ultra-Wideband and Computer Vision. Animals 2025, 15, 456. https://doi.org/10.3390/ani15030456

20. Sohn, K. Improved Deep Metric Learning with Multi-class N-pair Loss Objective. In Advances in Neural Information Processing Systems; 2016, 29. Available online: https://papers.nips.cc/paper_files/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html (accessed on 16 September 2026).

21. Chen, T.; Kornblith, S.; Norouzi, M.; Hinton, G. A Simple Framework for Contrastive Learning of Visual Representations. In Proceedings of the 37th International Conference on Machine Learning; 2020, 119, 1597–1607. Available online: https://proceedings.mlr.press/v119/chen20j.html (accessed on 16 September 2026).

22. Schroff, F.; Kalenichenko, D.; Philbin, J. FaceNet: A unified embedding for face recognition and clustering. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR); 2015, 815–823. https://doi.org/10.1109/CVPR.2015.7298682

23. Movshovitz-Attias, Y.; Toshev, A.; Leung, T.K.; Ioffe, S.; Singh, S. No Fuss Distance Metric Learning Using Proxies. In 2017 IEEE International Conference on Computer Vision (ICCV); 2017, 360–368. https://doi.org/10.1109/ICCV.2017.47

24. He, K.; Zhang, X.; Ren, S.; Sun, J. Deep Residual Learning for Image Recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR); 2016, 770–778. https://doi.org/10.1109/CVPR.2016.90

25. Huang, G.; Liu, Z.; Van Der Maaten, L.; Weinberger, K.Q. Densely Connected Convolutional Networks. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR); 2017, 2261–2269. https://doi.org/10.1109/cvpr.2017.243

26. Geirhos, R.; Jacobsen, J.H.; Michaelis, C.; Zemel, R.; Brendel, W.; Bethge, M.; Wichmann, F.A. Shortcut learning in deep neural networks. Nat. Mach. Intell. 2020, 2, 665–673. https://doi.org/10.1038/s42256-020-00257-z

27. Roberts, D.R.; Bahn, V.; Ciuti, S.; Boyce, M.S.; Elith, J.; Guillera‐Arroita, G.; Hauenstein, S.; Lahoz‐Monfort, J.J.; Schröder, B.; Thuiller, W.; Warton, D.I.; Wintle, B.A.; Hartig, F.; Dormann, C.F. Cross‐validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 2017, 40, 913–929. https://doi.org/10.1111/ecog.02881

28. Cawley, G.C.; Talbot, N.L.C. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. J. Mach. Learn. Res. 2010, 11, 2079–2107. Available online: https://www.jmlr.org/papers/v11/cawley10a.html (accessed on 16 September 2026).

29. Koh, P.W.; Sagawa, S.; Marklund, H.; Xie, S.M.; Zhang, M.; Balsubramani, A.; Hu, W.; Yasunaga, M.; Phillips, R.L.; Gao, I.; Lee, T.; David, E.; Stavness, I.; Guo, W.; Earnshaw, B.; Haque, I.; Beery, S.M.; Leskovec, J.; Kundaje, A.; Pierson, E.; Levine, S.; Finn, C.; Liang, P. WILDS: A Benchmark of in-the-Wild Distribution Shifts. In Proceedings of the 38th International Conference on Machine Learning; 2021, 139, 5637–5664. Available online: https://proceedings.mlr.press/v139/koh21a.html (accessed on 16 September 2026).

30. Möller, S.; Hölscher, M.; Morisse, K. Automatisierte Erzeugung eines Trainingsdatensatzes zur bildbasierten Tieridentifikation mittels KI. In Digitale Infrastrukturen, Lecture Notes in Informatics; 2025, 358, 351–356. https://doi.org/10.18420/giljt2025_41

31. Russakovsky, O.; Deng, J.; Su, H.; Krause, J.; Satheesh, S.; Ma, S.; Huang, Z.; Karpathy, A.; Khosla, A.; Bernstein, M.; Berg, A.C.; Fei-Fei, L. ImageNet Large Scale Visual Recognition Challenge. Int. J. Comput. Vis. 2015, 115, 211–252. https://doi.org/10.1007/s11263-015-0816-y

32. Efron, B. Bootstrap Methods: Another Look at the Jackknife. Ann. Stat. 1979, 7, 1–26. https://doi.org/10.1214/aos/1176344552
