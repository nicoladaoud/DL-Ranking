# DL-Ranking
Using Deep Learning methods (paragraph vectors) to Rank documents against a query in lucene index. For the purpose of this code, we use a RAMDirectory since we are storing a small amount of dummy data. For larger amounts of data, we will use a regular directory instead. For the sake of this example, we are ranking pizza sentences against a search query. Some of the code examples have been used from the cited book below.



##Citations/Sources:
@misc{le2014distributed,
    title={Distributed Representations of Sentences and Documents},
    author={Quoc V. Le and Tomas Mikolov},
    year={2014},
    eprint={1405.4053},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

Teofili, Tommaso, and Chris Mattmann. Deep Learning for Search. Manning Publications Co., 2019. 
