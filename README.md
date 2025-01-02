# Algorithmic Curation of Inclusive Syllabi: 
## Tool for Measuring and Recommending Diversity in Course Content

### Relevant Code
The main code for this project is stored in the following files. Other can may be supplemental, data, or for testing purposes.

`backend/pipeline.py` - Main pipeline for processing syllabi and generating recommendations.
`interface/` - Frontend interface for uploading syllabi and viewing recommendations.

### Abstract
Diversity in university syllabi fosters student engagement, enhances critical thinking, and ensures equitable representation of perspectives, preparing students for a globalized world. However, curating diverse syllabi presents significant challenges, including the time-intensive task of identifying inclusive materials and the lack of institutional support or incentives for regular syllabus revision. To address these issues, we introduce a framework that integrates quantitative metrics—such as Rao’s Entropy, Jaccard’s Distance, and relevance and overlap proportions—to evaluate and enhance syllabus diversity. By leveraging the Open Library API, this framework automates metadata retrieval and subject analysis, streamlining the process and reducing reliance on manual efforts. Empirical results suggest that the framework, when paired with an appropriate metric, effectively measures thematic richness and dissimilarity while providing actionable recommendations for diversifying syllabus content. Beyond higher education, the framework could support applications such as designing inclusive patient scenarios for medical training, curating diverse datasets for algorithmic fairness, and selecting representative case studies for business education. These applications highlight its potential to foster diversity across a variety of fields and contexts.
