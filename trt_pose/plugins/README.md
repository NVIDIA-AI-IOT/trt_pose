## TODO

- [x] find_peaks plain cpp
- [x] find_peaks torch binding
- [x] refine_peaks plain cpp
- [x] refine_peaks torch binding
- [x] paf_score_graph plain cpp
- [x] paf_score_graph torch binding
- [x] munkres plain cpp
- [x] munkres torch binding
- [x] connect parts plain cpp
- [x] connect parts torch binding
- [x] test full refactored pipeline

## Terminology

* ``N`` - int - Batch size
* ``C`` - int - Number of part types
* ``K`` - int - Number of linkage types
* ``M`` - int - Max number of parts per part type
* ``O`` - int - Max number of objects per image
* ``H`` - int - cmap/paf height
* ``W`` - int - cmap/paf width
* ``cmap`` - float - ``NxCxHxW`` - Part confidence maps
* ``paf`` - float - ``Nx(2K)xHxW`` - Part affinity fields
* ``peaks`` - int - ``NxCxMx2`` - i, j coordinates of peaks in confidence map in range ``[0, H-1], [0, W-1]``
* ``peak_counts`` - int - ``NxC`` - number of peaks per part type
* ``normalized_peaks`` - float - ``NxCxMx2`` - refined and normalized i, j coordinates in range ``[0, 1]``
* ``topology`` - int - ``Kx4`` - topology of linkages, each entry has ``[k_i, k_j, c_a, c_b]``, where ``k_i, k_j`` are the channels in paf corresponding to ``i, j`` dimensions and ``c_a`` is the source part type and ``c_b`` is the sink part type
* ``score_graph`` - float - ``NxKxMxM`` - the linkage score graph computed by evaluating line integrals between part pairs of each linkage type
* ``objects`` - int - ``NxOxC - objects parsed, each entry is -1 if the part is not present, otherwise the entry is the index of the part (for it's part type)
* ``object_counts`` - int - ``N`` - the object count for each image
* ``connections`` - int - ``NxKx2xM`` - the connection graph.

ie, for link type ``1`` and source part index ``5``, the sink part index ``3`` = ``connections[0][1][0][5]``

or for link type ``1`` and sink part index ``3`` the source part index ``5`` is
``connections[0][1][1][3]``

the graph representation is redundant, both directions are included for constant time lookup.  if no connection exists for a sink or source part index, it will return ``-1``

## Post Processing

### find_peaks

```python
peak_counts, peaks = find_peaks(cmap, threshold=0.1, window_size=3, max_count=100)
```

### refine_peaks

```python
normalized_peaks = refine_peaks(peak_counts, peaks, cmap, window_size=3)
```
### paf_score_graph

```python
score_graph = paf_score_graph(paf, topology, peak_counts, normalized_peaks, num_integral_samples=5)
```

### assignment


```python
connections = assignment(score_graph, topology, peak_counts, score_threshold=0.1)
```

### connect_parts


```python
object_counts, objects = connect_parts(connections, topology, peak_counts, max_count=100)
```

## Training

### generate_cmap

```python
cmap = generate_cmap(peak_counts, normalized_peaks, height=46, width=46, stdev=1, window=9)
```

### generate_paf


```python
paf = generate_paf(connections, topology, peak_counts, normalized_peaks, height=46, width=46, stdev=1)
```
