# Approximate Multiparametric Mixed-integer Convex Programming

<p align="center">
	<img width="500" src="/figures/readme_image.png?raw=true">
</p>
<p align="center" width="600">
Figure: control evaluation time. Bars show the mean while error bars shown the
minimum and maximum values. The explicit implementation is up to three orders of
magnitude faster than on-line optimization.
</p>

## General Description

This repository implements the algorithm for generatic suboptimal explicit
solutions of multiparametric mixed-integer convex programs, submitted to [IEEE
Control Systems Letters](http://ieee-cssletters.dei.unipd.it/index.php). The
algorithm can be run either locally or on a cluster via `mpirun`.

``` 
@ARTICLE{Mayuta2019,
       author = {{Malyuta}, Danylo and {A\c{c}{\i}kme\c{s}e}, Beh\c{c}et},
        title = {Approximate Multiparametric Mixed-integer Convex Programming},
      journal = {arXiv e-prints},
     keywords = {Mathematics - Optimization and Control},
         year = "2019",
        month = "Feb",
          eid = {arXiv:1902.10994},
        pages = {arXiv:1902.10994},
archivePrefix = {arXiv},
       eprint = {1902.10994},
 primaryClass = {math.OC},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190210994M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Requirements

To run the code, you must have Python 3.7.2 and [MOSEK
9.0.87](https://www.mosek.com/downloads/) installed. To install Python and other
dependenies (except MOSEK) on Ubuntu, we recommend that you install [Anaconda
for Python 3.7](https://www.anaconda.com/distribution/) and then execute (from
inside this repository's directory):

```
$ conda create -n py372 python=3.7.2 anaconda # Answer yes to everything
$ source activate py372
$ pip install -r requirements.txt
```

## Instructions

Partitioning jobs are created through `make_jobs.sh`. Run

```
bash make_jobs.sh -h
```
for more information. The job files are stored in the `./runtime` directory.
