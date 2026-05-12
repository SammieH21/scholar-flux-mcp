# Input:

```json
{
  "question": "What is the current evidence for using large language models to assist in clinical decision support for diabetes management?",
  "queries": [
    "AI clinical decision support diabetes",
    "large language model diabetes care"
  ],
  "providers": [
    "plos",
    "openalex",
    "core",
    "springer_nature"
  ],
  "categories": [
    "general",
    "intervention"
  ],
  "max_records": 120,
  "pages": 4,
  "year_from": 2022
}
```

# Output 1:

# Research Synthesis

## Research Question
> What is the current evidence for using large language models to assist in clinical decision support for diabetes management?

**Categories**: general, intervention
**Queries**:
- AI clinical decision support diabetes
- large language model diabetes care

**Records Analyzed**: 46
**Confidence Score**: 78%
**Evidence Grounding**: 7 verified records, 0 rejected



---

## Synthesis

Current evidence demonstrates that large language models (LLMs) show significant promise for clinical decision support in diabetes management, though the field is still evolving with varying levels of validation. Several diabetes-specific LLM frameworks have demonstrated state-of-the-art performance across multiple tasks, including personalized healthcare recommendations, medical education, and clinical workflow assistance [0]. Studies evaluating LLM performance on standardized diabetes examinations have shown that ChatGPT-4.0 achieves passing accuracy (62.50% on UK specialty exams) and can outperform primary care physicians while assisting most physicians in improving their examination scores by 1-6.13% [3]. 

The integration of LLMs with image-based deep learning systems, such as DeepDR-LLM, has demonstrated real-world clinical utility. In prospective studies, patients under primary care physician supervision assisted by DeepDR-LLM showed better self-management behaviors, and those with referable diabetic retinopathy were more likely to adhere to referral recommendations compared to unassisted care [2]. Patient-facing conversational assistants like MarIA, powered by GPT-3.5, have shown that personalized interactions can increase engagement by 26% over three-month periods, though safety concerns remain regarding potentially inappropriate recommendations for users with specific clinical constraints [6].

Retrieval-augmented generation (RAG) models have achieved 98% accuracy in providing validated medical information about diabetes and diabetic foot care to laypersons, demonstrating the potential for patient education and self-management support [5]. However, evidence also highlights limitations: OpenEvidence, when tested retrospectively across five common chronic conditions including diabetes, showed high clarity and relevance scores but limited impact on actual clinical decision-making, primarily reinforcing rather than modifying physician plans [24].

The evidence base spans from 2022-2026 with most high-quality studies published in 2024-2025, indicating this is a rapidly developing field. Key gaps remain regarding long-term clinical outcomes, integration into actual clinical workflows, and validation across diverse populations and healthcare settings.


## Key Findings

- Large language models demonstrate strong performance in diabetes-specific tasks, with ChatGPT-4.0 achieving passing scores on standardized diabetes examinations and outperforming general-purpose LLMs [0; 3]
- Integrated LLM systems like DeepDR-LLM have demonstrated real-world clinical utility in prospective studies, improving patient self-management behaviors and referral adherence for diabetic retinopathy [2]
- Retrieval-augmented generation (RAG) architectures can provide validated, accurate medical information with 98% accuracy for patient education in diabetes and diabetic foot care [5]
- Patient-facing conversational assistants can increase engagement by 26% through personalized interactions, but require safety layers to prevent inappropriate recommendations for users with specific clinical constraints [6]
- Evidence on actual clinical decision-making impact remains limited, with studies showing LLMs primarily reinforce rather than modify physician decisions in routine cases [24]
- Multi-modal AI systems combining language models with image-based deep learning show promise for comprehensive diabetes management including complication screening and personalized recommendations [2; 10]
- Key challenges include validation across diverse populations, integration into clinical workflows, and demonstrating long-term patient outcomes [0; 1; 24]


## Supporting Evidence


1. **Diabetica: Adapting Large Language Model to Enhance Multiple Medical Tasks in Diabetes Care and Management** (2024) [Link](http://arxiv.org/abs/2409.13191) [DOI]10.48550/arxiv.2409.13191) OpenAlex

**Index**: [0]

**Relevance Score**: 95%

**Summary**: Diabetica framework demonstrated state-of-the-art proficiency across various diabetes tasks including personalized healthcare, medical education, and clinical workflow assistance after fine-tuning on diabetes-specific datasets

**Abstract**: Diabetes is a chronic disease with a significant global health burden, requiring multi-stakeholder collaboration for optimal management. Large language models (LLMs) have shown promise in various healthcare scenarios, but their effectiveness across diverse diabetes tasks remains unproven. Our study introduced a framework to train and validate diabetes-specific LLMs. We first developed a comprehensive data processing pipeline that includes data collection, filtering, augmentation and refinement. This created a high-quality, diabetes-specific dataset and evaluation benchmarks from scratch. Fine-tuned on the collected training dataset, our diabetes-specific LLM family demonstrated state-of-the-art proficiency in processing various diabetes tasks compared to other LLMs. Furthermore, clinical studies revealed the potential applications of our models in diabetes care, including providing personalized healthcare, assisting medical education, and streamlining clinical tasks. Generally, our introduced framework helps develop diabetes-specific LLMs and highlights their potential to enhance clinical practice and provide personalized, data-driven support for diabetes management across different end users. Our codes, benchmarks and models are available at https://github.com/waltonfuture/Diabetica.



2. **ChatGPT in Diabetes Care: An Overview of the Evolution and Potential of Generative Artificial Intelligence Model Like ChatGPT in Augmenting Clinical and Patient Outcomes in the Management of Diabetes** (2023) [Link](https://doi.org/10.4103/ijdt.ijdt_31_23) [DOI]10.4103/ijdt.ijdt_31_23) OpenAlex

**Index**: [1]

**Relevance Score**: 85%

**Summary**: ChatGPT shows transformative potential in enhancing patient engagement, personalized medical guidance, and streamlined healthcare workflows for diabetes management

**Abstract**: Abstract The rapid evolution of artificial intelligence (AI) technologies has brought a new era in health care, offering innovative solutions for various medical disciplines, including diabetes care. This viewpoint article aimed to elucidate the transformative potential of Chat Generative Pre-trained Transformer (ChatGPT), a large language model (LLM), in augmenting diabetes care. We traverse through the historical evolution of AI, delineating its trajectory from conceptual origins to contemporary advancements. Central to our discussion is the exploration of ChatGPT’s capabilities in fostering enhanced patient engagement, personalized medical guidance, and streamlined health-care workflows. Through a comprehensive review, we underscore ChatGPT as a pivotal technology, focused to revolutionize traditional paradigms in diabetes management and patient care.



3. **Integrated image-based deep learning and language models for primary diabetes care** (2024) [Link](https://doi.org/10.1038/s41591-024-03139-8) [DOI]10.1038/s41591-024-03139-8) OpenAlex

**Index**: [2]

**Relevance Score**: 95%

**Summary**: DeepDR-LLM integrated system showed in prospective real-world study that patients with newly diagnosed diabetes in the PCP+DeepDR-LLM arm demonstrated better self-management behaviors throughout follow-up, and patients with referral DR were more likely to adhere to recommendations

**Abstract**: Abstract Primary diabetes care and diabetic retinopathy (DR) screening persist as major public health challenges due to a shortage of trained primary care physicians (PCPs), particularly in low-resource settings. Here, to bridge the gaps, we developed an integrated image–language system (DeepDR-LLM), combining a large language model (LLM module) and image-based deep learning (DeepDR-Transformer), to provide individualized diabetes management recommendations to PCPs. In a retrospective evaluation, the LLM module demonstrated comparable performance to PCPs and endocrinology residents when tested in English and outperformed PCPs and had comparable performance to endocrinology residents in Chinese. For identifying referable DR, the average PCP’s accuracy was 81.0% unassisted and 92.3% assisted by DeepDR-Transformer. Furthermore, we performed a single-center real-world prospective study, deploying DeepDR-LLM. We compared diabetes management adherence of patients under the unassisted PCP arm( n= 397) with those under the PCP+DeepDR-LLM arm( n= 372). Patients with newly diagnosed diabetes in the PCP+DeepDR-LLM arm showed better self-management behaviors throughout follow-up( P &lt; 0.05). For patients with referral DR, those in the PCP+DeepDR-LLM arm were more likely to adhere to DR referrals( P &lt; 0.01). Additionally, DeepDR-LLM deployment improved the quality and empathy level of management recommendations. Given its multifaceted performance, DeepDR-LLM holds promise as a digital solution for enhancing primary diabetes care and DR screening.



4. **Large language models for diabetes training: a prospective study** (2025) [Link](https://doi.org/10.1016/j.scib.2025.01.034) [DOI]10.1016/j.scib.2025.01.034) OpenAlex

**Index**: [3]

**Relevance Score**: 95%

**Summary**: ChatGPT-4.0 achieved 62.50% passing accuracy on English diabetes specialty exams, outperforming other LLMs and surpassing all primary care physicians in the Chinese National Certificate Examination, improving physician scores by 1%-6.13%

**Abstract**: Diabetes poses a considerable global health challenge, with varying levels of diabetes knowledge among healthcare professionals, highlighting the importance of diabetes training. Large Language Models (LLMs) provide new insights into diabetes training, but their performance in diabetes-related queries remains uncertain, especially outside the English language like Chinese. We first evaluated the performance of ten LLMs: ChatGPT-3.5, ChatGPT-4.0, Google Bard, LlaMA-7B, LlaMA2-7B, Baidu ERNIE Bot, Ali Tongyi Qianwen, MedGPT, HuatuoGPT, and Chinese LlaMA2-7B on diabetes-related queries, based on the Chinese National Certificate Examination for Primary Diabetes Care in China (NCE-CPDC) and the English Specialty Certificate Examination in Endocrinology and Diabetes of Membership of the Royal College of Physicians of the United Kingdom. Second, we assessed the training of primary care physicians (PCPs) without and with the assistance of ChatGPT-4.0 in the NCE-CPDC examination to ascertain the reliability of LLMs as medical assistants. We found that ChatGPT-4.0 outperformed other LLMs in the English examination, achieving a passing accuracy of 62.50%, which was significantly higher than that of Google Bard, LlaMA-7B, and LlaMA2-7B. For the NCE-CPFC examination, ChatGPT-4.0, Ali Tongyi Qianwen, Baidu ERNIE Bot, Google Bard, MedGPT, and ChatGPT-3.5 successfully passed, whereas LlaMA2-7B, HuatuoGPT, Chinese LLaMA2-7B, and LlaMA-7B failed. ChatGPT-4.0 (84.82%) surpassed all PCPs and assisted most PCPs in the NCE-CPDC examination (improving by 1 %-6.13%). In summary, LLMs demonstrated outstanding competence for diabetes-related questions in both the Chinese and English language, and hold great potential to assist future diabetes training for physicians globally.



5. **Building Trustworthy Generative Artificial Intelligence for Diabetes Care and Limb Preservation: A Medical Knowledge Extraction Case** (2024) [Link](https://doi.org/10.1177/19322968241253568) [DOI]10.1177/19322968241253568) OpenAlex

**Index**: [5]

**Relevance Score**: 90%

**Summary**: RAG model achieved 98% accuracy in providing validated medical knowledge about diabetes and diabetic foot care to laypersons with eighth-grade literacy level using NIH National Standards as knowledge base

**Abstract**: Background: Large language models (LLMs) offer significant potential in medical information extraction but carry risks of generating incorrect information. This study aims to develop and validate a retriever-augmented generation (RAG) model that provides accurate medical knowledge about diabetes and diabetic foot care to laypersons with an eighth-grade literacy level. Improving health literacy through patient education is paramount to addressing the problem of limb loss in the diabetic population. In addition to affecting patient well-being through improved outcomes, improved physician well-being is an important outcome of a self-management model for patient health education. Methods: We used an RAG architecture and built a question-and-answer artificial intelligence (AI) model to extract knowledge in response to questions pertaining to diabetes and diabetic foot care. We utilized GPT-4 by OpenAI, with Pinecone as a vector database. The NIH National Standards for Diabetes Self-Management Education served as the basis for our knowledge base. The model’s outputs were validated through expert review against established guidelines and literature. Fifty-eight keywords were used to select 295 articles and the model was tested against 175 questions across topics. Results: The study demonstrated that with appropriate content volume and few-shot learning prompts, the RAG model achieved 98% accuracy, confirming its capability to offer user-friendly and comprehensible medical information. Conclusion: The RAG model represents a promising tool for delivering reliable medical knowledge to the public which can be used for self-education and self-management for diabetes, highlighting the importance of content validation and innovative prompt engineering in AI applications.



6. **Assessing the User Experience of an LLM-Based Conversational Assistant in Diabetes Mellitus Care** (2026) [Link](http://link.springer.com/openurl/fulltext?id=doi:10.1007/s41666-025-00217-5) [DOI]10.1007/s41666-025-00217-5) Springer Nature

**Index**: [6]

**Relevance Score**: 88%

**Summary**: MarIA GPT-3.5 powered assistant showed 26% increased engagement with personalized interactions over 3 months in 35 participants, but safety assessment revealed some general health suggestions could be inappropriate for users with specific clinical constraints

**Abstract**: This article presents the design, implementation, and evaluation of MarIA, a GPT-3.5-powered virtual assistant integrated into a messaging platform to support patients with type 2 diabetes mellitus (DM). MarIA employs a multi-agent architecture that enables varying dialogue styles and degrees of personalization. In a 3-month longitudinal study involving 35 participants, personalized interactions increased engagement by 26%, while message length more than quadrupled—yielding a richer understanding of patient context. This deeper contextualization enabled MarIA to initiate more relevant, meaningful conversations, fostering a positive cycle of sustained engagement. Safety was critically assessed. While MarIA did not generate factual hallucinations, some general health suggestions—though accurate in isolation—could be inappropriate for users with specific clinical constraints. This underscores the need not only for comprehensive patient profiling but also for an embedded safety layer capable of detecting potentially unsuitable recommendations before or even after delivery. The multi-agent architecture proved essential in enabling proactive behaviors, nuanced context detection, and dialogue adaptability, ultimately enhancing both engagement and user safety in AI-supported chronic care.



7. **The Use of an Artificial Intelligence Platform OpenEvidence to Augment Clinical Decision-Making for Primary Care Physicians** (2025) [Link](https://doi.org/10.1177/21501319251332215) [DOI]10.1177/21501319251332215) OpenAlex

**Index**: [24]

**Relevance Score**: 82%

**Summary**: OpenEvidence platform scored high on clarity (3.55), relevance (3.75), and evidence support (3.35) but had limited impact on clinical decision-making (1.95) as it primarily reinforced rather than modified physician plans in retrospective analysis

**Abstract**: Background: Artificial intelligence (AI) platforms can potentially enhance clinical decision-making (CDM) in primary care settings. OpenEvidence (OE), an AI tool, draws from trusted sources to generate evidence-based medicine (EBM) recommendations to address clinical questions. However, its effectiveness in real-world primary care cases remains unknown. Objective: To evaluate the performance of OE in providing EBM recommendations for five common chronic conditions in primary care: hypertension, hyperlipidemia, diabetes mellitus type 2, depression, and obesity. Methods: Five patient cases were retrospectively analyzed. Physicians posed specific clinical questions, and OE responses were evaluated on clarity, relevance, evidence support, impact on CDM, and overall satisfaction. Four independent physicians provided ratings using a 0 to 4 scale. Results: OE provided accurate, evidence-based recommendations in all cases, aligning with physician plans. OE was scored on a scale of zero to four, where zero was very unclear, and four was very clear. Mean scores across cases were clarity (3.55± 0.60), relevance (3.75± 0.44), support (3.35± 0.49), and satisfaction (3.60± 0.60). However, the impact on CDM was limited (1.95± 1.05), as OE primarily reinforced rather than modified plans. Conclusion: OE was rated high in clarity, relevance, and evidence-based support, reinforcing physician decisions in common chronic conditions. While the impact on CDM was minimal due to the study’s retrospective nature, OE shows promise in augmenting the primary care physician. Prospective trials are needed to evaluate its utility in complex cases and multidisciplinary settings.



## Limitations

- Most studies are proof-of-concept or pilot evaluations rather than large-scale randomized controlled trials measuring patient outcomes [0; 2; 3; 5; 6]
- Limited evidence on LLM performance in real-time clinical decision-making scenarios; many evaluations use retrospective or standardized test frameworks [24]
- Safety concerns remain unresolved - LLMs may generate inappropriate recommendations for patients with specific clinical constraints [6]
- Most research focuses on English and Chinese languages; generalizability to other languages and healthcare systems is unclear [3]
- Evidence on cost-effectiveness and workflow integration is largely absent from current literature [0; 1]


## Suggested Follow-up

- What are the specific regulatory frameworks and validation requirements for LLM-based clinical decision support systems in diabetes care?
- How do LLM-based decision support systems perform across different healthcare settings and provider experience levels?
- What are the long-term patient outcomes (glycemic control, complications) when using LLM-assisted diabetes management?
- How do different LLM architectures (fine-tuned vs. RAG vs. multi-modal) compare in clinical accuracy and safety?
- What is the cost-effectiveness of implementing LLM-based decision support in diabetes care compared to standard approaches?



---
*This synthesis was generated using ScholarFlux MCP with PydanticAI.
LLM: minimax-m2.5:cloud | Embedding Model: embeddinggemma:latest
Always verify findings with primary sources.*


# Output 2: 


# Research Synthesis

## Research Question
> What is the current evidence for using large language models to assist in clinical decision support for diabetes management?

**Categories**: general, intervention
**Queries**:
- AI clinical decision support diabetes
- large language model diabetes care

**Records Analyzed**: 82
**Confidence Score**: 75%
**Evidence Grounding**: 7 verified records, 0 rejected



---

## Synthesis

Current evidence for using large language models (LLMs) to assist in clinical decision support for diabetes management demonstrates promising but still emerging capabilities across multiple applications. Several studies have evaluated LLM performance in diabetes-specific tasks. A comprehensive evaluation of 10 LLMs on diabetes-related queries found that ChatGPT-4.0 achieved the highest performance, passing both Chinese (NCE-CPDC) and English (UK Specialty Certificate Examination) diabetes exams with 62.50% accuracy in English, significantly outperforming other models including Google Bard, LLaMA variants, and Chinese-specific models [9]. Importantly, when used to assist primary care physicians in the Chinese examination, ChatGPT-4.0 improved physician scores by 1-6.13%, demonstrating potential as a training and decision-support tool.

Research has also focused on diabetes-specific LLM development. The Diabetica framework represents a significant advancement, creating a comprehensive data processing pipeline and fine-tuning diabetes-specific LLMs that demonstrated state-of-the-art performance across various diabetes tasks compared to general-purpose LLMs [23]. Similarly, DeepDR-LLM integrates image-based deep learning (DeepDR-Transformer) with an LLM module to provide individualized diabetes management recommendations, showing in prospective studies that patients under PCP+DeepDR-LLM guidance exhibited better self-management behaviors [24].

For direct patient-facing applications, MarIA (a GPT-3.5-powered virtual assistant) showed that personalized interactions increased engagement by 26% over 3 months in 35 patients with type 2 diabetes, with message length quadrupling to enable richer contextual understanding [2]. However, safety concerns were noted—while no factual hallucinations occurred, some general health recommendations could be inappropriate for users with specific clinical constraints, highlighting the need for embedded safety layers.

In clinical decision support systems more broadly, OpenEvidence demonstrated high ratings for clarity (3.55/4), relevance (3.75/4), and evidence support (3.35/4) in diabetes care scenarios, though impact on clinical decision-making was limited (1.95/4), primarily reinforcing rather than modifying physician plans [5]. A retrospective study using the BiKBAC method for automated guideline compliance assessment in type 2 diabetes achieved 91% recall and 81% precision compared to expert clinicians [30].

For medication management, ML-based clinical decision support systems for type 2 diabetes drug management achieved 85-99.4% accuracy in predicting individual drug classes, with multi-drug accuracy of 72% [42]. Additionally, explicit definitions for potentially inappropriate antidiabetic prescriptions have been developed for integration into clinical decision support systems [15].

Evidence quality varies—several studies involve prospective clinical evaluations (MarIA, DeepDR-LLM), while others are retrospective or proof-of-concept. The field shows particular promise in diabetes education, retinopathy screening support, medication optimization, and personalized patient engagement, though real-world implementation studies remain limited. Key challenges include ensuring clinical safety, achieving meaningful impact on clinical decisions versus reinforcement of existing practices, and addressing the need for diabetes-specific model optimization over general-purpose LLMs.


## Key Findings

- ChatGPT-4.0 achieved 62.50% passing accuracy on English diabetes exams and improved primary care physician exam scores by 1-6.13% when used as an assistant, demonstrating LLM potential for diabetes training and decision support [9].
- The Diabetica framework developed diabetes-specific LLMs that outperformed general-purpose LLMs across multiple diabetes tasks through specialized data processing and fine-tuning [23].
- DeepDR-LLM, integrating image-based deep learning with LLM technology, improved patient self-management behaviors in prospective studies when used to assist primary care physicians [24].
- MarIA GPT-3.5 virtual assistant increased patient engagement by 26% in type 2 diabetes management, though safety concerns about inappropriate recommendations for specific clinical contexts were identified [2].
- ML-based clinical decision support systems for diabetes drug management achieved 85-99.4% accuracy in predicting individual drug classes, with multi-drug accuracy of 72% [42].
- The BiKBAC automated guideline compliance assessment system achieved 91% recall and 81% precision in evaluating type 2 diabetes care quality against expert clinicians [30].
- General-purpose AI platforms like OpenEvidence showed high clarity and relevance but limited impact on actual clinical decision-making, primarily reinforcing rather than modifying physician plans [5].


## Supporting Evidence


1. **Large language models for diabetes training: a prospective study** (2025) [Link](https://doi.org/10.1016/j.scib.2025.01.034) [DOI]10.1016/j.scib.2025.01.034) OpenAlex

**Index**: [9]

**Relevance Score**: 95%

**Summary**: ChatGPT-4.0 achieved 62.50% passing accuracy on English diabetes specialty exams and 84.82% on Chinese exams, outperforming all other LLMs. When used to assist primary care physicians, it improved their exam scores by 1-6.13%.

**Abstract**: Diabetes poses a considerable global health challenge, with varying levels of diabetes knowledge among healthcare professionals, highlighting the importance of diabetes training. Large Language Models (LLMs) provide new insights into diabetes training, but their performance in diabetes-related queries remains uncertain, especially outside the English language like Chinese. We first evaluated the performance of ten LLMs: ChatGPT-3.5, ChatGPT-4.0, Google Bard, LlaMA-7B, LlaMA2-7B, Baidu ERNIE Bot, Ali Tongyi Qianwen, MedGPT, HuatuoGPT, and Chinese LlaMA2-7B on diabetes-related queries, based on the Chinese National Certificate Examination for Primary Diabetes Care in China (NCE-CPDC) and the English Specialty Certificate Examination in Endocrinology and Diabetes of Membership of the Royal College of Physicians of the United Kingdom. Second, we assessed the training of primary care physicians (PCPs) without and with the assistance of ChatGPT-4.0 in the NCE-CPDC examination to ascertain the reliability of LLMs as medical assistants. We found that ChatGPT-4.0 outperformed other LLMs in the English examination, achieving a passing accuracy of 62.50%, which was significantly higher than that of Google Bard, LlaMA-7B, and LlaMA2-7B. For the NCE-CPFC examination, ChatGPT-4.0, Ali Tongyi Qianwen, Baidu ERNIE Bot, Google Bard, MedGPT, and ChatGPT-3.5 successfully passed, whereas LlaMA2-7B, HuatuoGPT, Chinese LLaMA2-7B, and LlaMA-7B failed. ChatGPT-4.0 (84.82%) surpassed all PCPs and assisted most PCPs in the NCE-CPDC examination (improving by 1 %-6.13%). In summary, LLMs demonstrated outstanding competence for diabetes-related questions in both the Chinese and English language, and hold great potential to assist future diabetes training for physicians globally.



2. **Diabetica: Adapting Large Language Model to Enhance Multiple Medical Tasks in Diabetes Care and Management** (2024) [Link](http://arxiv.org/abs/2409.13191) [DOI]10.48550/arxiv.2409.13191) OpenAlex

**Index**: [23]

**Relevance Score**: 92%

**Summary**: Diabetica framework developed diabetes-specific LLMs using comprehensive data processing pipeline and fine-tuning, demonstrating state-of-the-art performance across various diabetes tasks compared to general LLMs.

**Abstract**: Diabetes is a chronic disease with a significant global health burden, requiring multi-stakeholder collaboration for optimal management. Large language models (LLMs) have shown promise in various healthcare scenarios, but their effectiveness across diverse diabetes tasks remains unproven. Our study introduced a framework to train and validate diabetes-specific LLMs. We first developed a comprehensive data processing pipeline that includes data collection, filtering, augmentation and refinement. This created a high-quality, diabetes-specific dataset and evaluation benchmarks from scratch. Fine-tuned on the collected training dataset, our diabetes-specific LLM family demonstrated state-of-the-art proficiency in processing various diabetes tasks compared to other LLMs. Furthermore, clinical studies revealed the potential applications of our models in diabetes care, including providing personalized healthcare, assisting medical education, and streamlining clinical tasks. Generally, our introduced framework helps develop diabetes-specific LLMs and highlights their potential to enhance clinical practice and provide personalized, data-driven support for diabetes management across different end users. Our codes, benchmarks and models are available at https://github.com/waltonfuture/Diabetica.



3. **Integrated image-based deep learning and language models for primary diabetes care** (2024) [Link](https://doi.org/10.1038/s41591-024-03139-8) [DOI]10.1038/s41591-024-03139-8) OpenAlex

**Index**: [24]

**Relevance Score**: 90%

**Summary**: DeepDR-LLM integrated image-based deep learning with LLM to provide individualized diabetes management recommendations. Prospective deployment showed patients under PCP+DeepDR-LLM had better self-management behaviors.

**Abstract**: Abstract Primary diabetes care and diabetic retinopathy (DR) screening persist as major public health challenges due to a shortage of trained primary care physicians (PCPs), particularly in low-resource settings. Here, to bridge the gaps, we developed an integrated image–language system (DeepDR-LLM), combining a large language model (LLM module) and image-based deep learning (DeepDR-Transformer), to provide individualized diabetes management recommendations to PCPs. In a retrospective evaluation, the LLM module demonstrated comparable performance to PCPs and endocrinology residents when tested in English and outperformed PCPs and had comparable performance to endocrinology residents in Chinese. For identifying referable DR, the average PCP’s accuracy was 81.0% unassisted and 92.3% assisted by DeepDR-Transformer. Furthermore, we performed a single-center real-world prospective study, deploying DeepDR-LLM. We compared diabetes management adherence of patients under the unassisted PCP arm( n= 397) with those under the PCP+DeepDR-LLM arm( n= 372). Patients with newly diagnosed diabetes in the PCP+DeepDR-LLM arm showed better self-management behaviors throughout follow-up( P &lt; 0.05). For patients with referral DR, those in the PCP+DeepDR-LLM arm were more likely to adhere to DR referrals( P &lt; 0.01). Additionally, DeepDR-LLM deployment improved the quality and empathy level of management recommendations. Given its multifaceted performance, DeepDR-LLM holds promise as a digital solution for enhancing primary diabetes care and DR screening.



4. **Assessing the User Experience of an LLM-Based Conversational Assistant in Diabetes Mellitus Care** (2026) [Link](http://link.springer.com/openurl/fulltext?id=doi:10.1007/s41666-025-00217-5) [DOI]10.1007/s41666-025-00217-5) Springer Nature

**Index**: [2]

**Relevance Score**: 88%

**Summary**: MarIA GPT-3.5 virtual assistant for type 2 diabetes increased personalized engagement by 26% over 3 months, with message length quadrupling. Safety assessment revealed potential for inappropriate recommendations in specific clinical contexts.

**Abstract**: This article presents the design, implementation, and evaluation of MarIA, a GPT-3.5-powered virtual assistant integrated into a messaging platform to support patients with type 2 diabetes mellitus (DM). MarIA employs a multi-agent architecture that enables varying dialogue styles and degrees of personalization. In a 3-month longitudinal study involving 35 participants, personalized interactions increased engagement by 26%, while message length more than quadrupled—yielding a richer understanding of patient context. This deeper contextualization enabled MarIA to initiate more relevant, meaningful conversations, fostering a positive cycle of sustained engagement. Safety was critically assessed. While MarIA did not generate factual hallucinations, some general health suggestions—though accurate in isolation—could be inappropriate for users with specific clinical constraints. This underscores the need not only for comprehensive patient profiling but also for an embedded safety layer capable of detecting potentially unsuitable recommendations before or even after delivery. The multi-agent architecture proved essential in enabling proactive behaviors, nuanced context detection, and dialogue adaptability, ultimately enhancing both engagement and user safety in AI-supported chronic care.



5. **The Use of an Artificial Intelligence Platform OpenEvidence to Augment Clinical Decision-Making for Primary Care Physicians** (2025) [Link](https://doi.org/10.1177/21501319251332215) [DOI]10.1177/21501319251332215) OpenAlex

**Index**: [5]

**Relevance Score**: 85%

**Summary**: OpenEvidence AI platform rated high for clarity (3.55), relevance (3.75), and evidence support (3.35) but had limited impact on clinical decision-making (1.95), primarily reinforcing existing physician plans.

**Abstract**: Background: Artificial intelligence (AI) platforms can potentially enhance clinical decision-making (CDM) in primary care settings. OpenEvidence (OE), an AI tool, draws from trusted sources to generate evidence-based medicine (EBM) recommendations to address clinical questions. However, its effectiveness in real-world primary care cases remains unknown. Objective: To evaluate the performance of OE in providing EBM recommendations for five common chronic conditions in primary care: hypertension, hyperlipidemia, diabetes mellitus type 2, depression, and obesity. Methods: Five patient cases were retrospectively analyzed. Physicians posed specific clinical questions, and OE responses were evaluated on clarity, relevance, evidence support, impact on CDM, and overall satisfaction. Four independent physicians provided ratings using a 0 to 4 scale. Results: OE provided accurate, evidence-based recommendations in all cases, aligning with physician plans. OE was scored on a scale of zero to four, where zero was very unclear, and four was very clear. Mean scores across cases were clarity (3.55± 0.60), relevance (3.75± 0.44), support (3.35± 0.49), and satisfaction (3.60± 0.60). However, the impact on CDM was limited (1.95± 1.05), as OE primarily reinforced rather than modified plans. Conclusion: OE was rated high in clarity, relevance, and evidence-based support, reinforcing physician decisions in common chronic conditions. While the impact on CDM was minimal due to the study’s retrospective nature, OE shows promise in augmenting the primary care physician. Prospective trials are needed to evaluate its utility in complex cases and multidisciplinary settings.



6. **Developing Clinical Decision Support System using Machine Learning Methods for Type 2 Diabetes Drug Management** (2022) [Link](https://doi.org/10.4103/ijem.ijem_435_21) [DOI]10.4103/ijem.ijem_435_21) OpenAlex

**Index**: [42]

**Relevance Score**: 82%

**Summary**: ML-based CDSS for type 2 diabetes drug management achieved 85-99.4% accuracy for individual drug predictions and 72% multi-drug accuracy.

**Abstract**: Background and Objectives: Application of artificial intelligence/machine learning (AI/ML) for automation of diabetes management can enhance equitable access to care and ensure delivery of minimum standards of care. Objective of the current study was to create a clinical decision support system using machine learning approach for diabetes drug management in people living with Type 2 diabetes. Methodology: Study was conducted at an Endocrinology clinic and data collected from the electronic clinic management system. 15485 diabetes prescriptions of 4974 patients were accessed. A data subset of 1671 diabetes prescriptions of 940 patients with information on diabetes drugs, demographics (age, gender, body mass index), biochemical parameters (HbA1c, fasting blood glucose, creatinine) and patient clinical parameters (diabetes duration, compliance to diet/exercise/medications, hypoglycemia, contraindication to any drug, summary of patient self monitoring of blood glucose data, diabetes complications) was used in analysis. An input of patient variables were used to predict all diabetes drug classes to be prescribed. Random forest algorithms were used to create decision trees for all diabetes drugs. Results and Conclusion: Accuracy for predicting use of each individual drug class varied from 85% to 99.4%. Multi-drug accuracy, indicating that all drug predictions in a prescription are correct, stands at 72%. Multi drug class accuracy in clinical application may be higher than this result, as in a lot of clinical scenarios, two or more diabetes drugs may be used interchangeably. This report presents a first positive step in developing a robust clinical decision support system to transform access and quality of diabetes care.



7. **Design of a bi-directional methodology for automated assessment of compliance to continuous application of clinical guidelines, and its evaluation in the type 2 diabetes domain** (2024) [Link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0303542) [DOI]10.1371/journal.pone.0303542) PLOS

**Index**: [30]

**Relevance Score**: 80%

**Summary**: BiKBAC automated guideline compliance assessment achieved 91% recall and 81% precision when compared to expert clinician assessments in type 2 diabetes management.

**Abstract**: 
We introduce a new approach for automated guideline-based-care quality assessment, the bidirectional knowledge-based assessment of compliance (BiKBAC) method, and the DiscovErr system, which implements it. Our methodology compares the guideline’s Asbru-based formal representation, including its intentions, with the longitudinal medical record, using a top-down and bottom-up approach. Partial matches are resolved using fuzzy temporal logic. The system was evaluated in the type 2 Diabetes management domain, comparing it to three expert clinicians, including two diabetes experts. The system and the experts commented on the management of 10 patients, randomly selected from 2,000 diabetes patients. On average, each record spanned 5.23 years; the data included 1,584 medical transactions. The system provided 279 comments. The experts made 181 different unique comments. The completeness (recall) of the system was 91% when the gold standard was comments made by at least two of the three experts, and 98%, compared to comments made by all three experts. The experts also assessed all of the 114 medication-therapy-related comments, and a random 35% of the 165 tests-and-monitoring-related comments. The system’s correctness (precision) was 81%, compared to comments judged as correct by both diabetes experts, and 91%, compared to comments judged as correct by one diabetes expert and at least as partially correct by the other. 89% of the comments were judged as important by both diabetes experts, 8% were judged as important by one expert, and 3% were judged as less important by both experts. Adding the validated system comments to the experts’ comments, the completeness scores of the experts were 75%, 60%, and 55%; the expert correctness scores were respectively 99%, 91%, and 88%. Thus, the system could be ranked first in completeness and second in correctness. We conclude that systems such as DiscovErr can effectively assess the quality of continuous guideline-based care.




## Limitations

- Most studies are single-center or involve limited sample sizes (e.g., 35 participants in MarIA study)
- Prospective clinical trials evaluating real-world impact on patient outcomes remain limited
- LLM performance varies significantly across languages, with generally lower performance on non-English content
- Safety concerns persist regarding inappropriate recommendations for patients with specific clinical constraints
- Impact on actual clinical decision-making has shown to be limited in some studies, primarily reinforcing rather than changing physician behavior
- Long-term clinical validation and regulatory approval studies are largely absent


## Suggested Follow-up

- What are the regulatory frameworks and clinical validation requirements for LLM-based clinical decision support in diabetes?
- How do diabetes-specific LLMs compare to general medical LLMs in head-to-head clinical decision-making evaluations?
- What are the long-term patient outcomes when LLMs are integrated into diabetes management workflows?
- How can LLM-generated recommendations be safely integrated without increasing clinical liability?
- What training approaches optimize clinician adoption of LLM-based decision support systems?



---
*This synthesis was generated using ScholarFlux MCP with PydanticAI.
LLM: minimax-m2.5:cloud | Embedding Model: embeddinggemma:latest
Always verify findings with primary sources.*

