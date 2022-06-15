Module createBaselineVideoSaliencyMaps
--------------------------------------
Erwan DAVID (2018) -- IPI, LS2N, Nantes, France

**Create baseline saliency maps needed by InfoGain. Not implemented.**

STIMLIST
-----
Module readBinarySalmap
-----------------------
Erwan DAVID (2018) -- IPI, LS2N, Nantes, France

**Example showing how to read binary saliency maps and videos.**

**Usage**: *Python3 readBinarySalmap.py \[Frame index\] \[Path to binary\]*



-----
Module scanpathMeasure
----------------------
Erwan DAVID (2018) -- IPI, LS2N, Nantes, France

**Scanpath maps/videos comparison tools and example as main**


## Functions

#### alignScanpaths(WMat)

    Compute shortest path in weight matrix from first elements to last elements of both scanpaths as part of the alignment procedure in Jarodzka's algorithm (MultiMatch).
    Dijkstra's shortest path algo

#### compareScanpath(fixations1, fixations2, starts1, starts2, iSC1, iSC2, weight=[1, 1])

    Return comparison scores between two scanpaths.
    Option to grapically display weight matrix, scanpath aligment, scanpaths vectors and final measures with matplotlib.

#### computeWeightMatrix(VEC1, VEC2, weight)

    Return weight matrix for alignment in Jarodzka's algorithm (MultiMatch).

#### dist_angle(vec1, vec2)

    Angle between two vectors - same result as orthodromic distance

#### dist_starttime(t1, t2)

    Different between fixation starting timestamp fitted with exponential

#### getScanpath(fixationList, startPositions, scanpathIdx=0)

    Return a scanpath in a list of scanpaths

#### getStartPositions(fixationList)

    Return positions of first fixation in list of scanpaths.
    Get starting indices of individual fixation sequences.

#### getValues(VEC1, VEC2, i1, i2)

    Measure distance and angle between all fixations that happened during a frame.
    Return scores normalized (0, 1). Lower is better.

#### sphre2UnitVector(sequence)

    Convert from longitude/latitude to 3D unit vectors
-----
Module saliencyMeasures
-----------------------
Erwan DAVID (2018) -- IPI, LS2N, Nantes, France

**Numpy implementations of Saliency maps/videos comparison tools (by Chencan Qian, Sep 2014 [*repo]) and example as main**

**Note**: *
  Numpy metrics are ported from Matlab implementation provided by http://saliency.mit.edu/<br />
  Bylinskii, Z., Judd, T., Durand, F., Oliva, A., & Torralba, A. (n.d.). MIT Saliency Benchmark.<br />
  Python/numpy implementation: Chencan Qian, Sep 2014 [*repo]<br />
  Python/pyorch implementation: Erwan David, 2018<br />
  [*repo] https://github.com/herrlich10/saliency<br />
  Refer to the MIT saliency Benchmark (website)[http://saliency.mit.edu/results_cat2000.html] for information about saliency measures<br />
*

## Functions

#### AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None)

    AUC_Borji

#### AUC_Judd(saliency_map, fixation_map, jitter=False)

    AUC_Judd

#### CC(saliency_map1, saliency_map2)

    Cross-Correlation (Pearson's linear coefficient)

#### InfoGain(saliency_map, fixation_map, baseline_map)

    InfoGain
    ref: Kümmerer, M., Wallis, T. S., & Bethge, M. (2015). Information-theoretic model comparison unifies saliency metrics. Proceedings of the National Academy of Sciences, 112(52), 16054-16059.
    repo matlab code: github.com/cvzoya/saliency/blob/master/code_forMetrics/InfoGain.m

#### KLD(p, q)

    Kullback-Leibler Divergence

#### NSS(saliency_map, fixation_map)

    Normalized Scanpath Saliency

#### SIM(saliency_map1, saliency_map2)

    SIMilarity measure (aka histogram intersection)

#### getSimVal(salmap1, salmap2, fixmap1=None, fixmap2=None, basemap=None)


#### get_binsalmap_info(filename)


#### match_hist(image, cdf, bin_centers, nbins=256)


#### normalize(x, method='standard', axis=None)

    Normalize data
    `standard`: i.e. z-score. Substract mean and divide by standard deviation

    `range`: normalize data to new bounds [0, 1]

    `sum`: normalize so that the sum of all element in tensor sum up to 1.

#### uniformSphereSampling(N)

    Equirectangular weighting by quasi-uniform over-sampling of a sphere.
    Used in Rai, Y., Gutiérrez, J., & Le Callet, P. (2017, June). A dataset of head and eye movements for 360 degree images. In Proceedings of the 8th ACM on Multimedia Systems Conference (pp. 205-210). ACM.
-----
