# An automated classification method for single cell RNA-Seq data based on the information content of individual genes
### Abstract

Single cell RNA sequencing (scRNA-seq) technologies promise to enable the quantitative study of biological processes at the single cell level [Patel, A. P. et al. 2014; Treutlein, B. et al 2014; Miyamoto, D. T. et al. 2015]. Commercial platforms such as 10x chromium are becoming established in lab practice [Hwang, B. et al., 2018; Dong, M. B. et al., 2019; Xiong, X. et al. 2019]. More than other high-throughput technologies, however, the reproducibility and accuracy of current analysis pipelines remains challenging [Kiselev, V. Y. et al., 2019]. For example, cellular classification algorithms continue to be evaluated using datasets with cell labels generated by computational analysis of transcriptomic data validated by the manual application of a priori biologic knowledge [Pouyan M.B. et al, 2018; Zhang et al, 2018; Jiang, H. 2019]. Thus, there is a crucial need for a benchmark that provides ground truth labels in an independent manner. Here, we develop such a benchmark using a dataset where ground truth labels are generated from surface protein level measurements. We demonstrate a substantial decrease in estimated accuracy of the current gold-standard, Seurat algorithm [Satija R. et al 2015, Butler, A. et al. 2018], in data with low information content. In order to overcome the challenge posed by noisy uninformative data, we implement an algorithm that optimizes information content through an information theory-based approach. Our approach yields a dramatic improvement in accuracy for a couple of clustering algorithms.

### Prerequisites

Please install all required packages listed in python_requirement.txt files using either pip or conda

```
pip3 install -r python_requirement.txt
```
or

```
conda install -r python_requirement.txt
```

Install all required R packages in the R_requirement.txt files using either bioconductor or CRAN

```
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install(R_requirement.txt)
```
or

```
install.packages(R_requirement.txt)
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

