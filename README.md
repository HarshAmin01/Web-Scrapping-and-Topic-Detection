# Introduction

Working with large language models locally, especially for tasks like zero-shot classification, can often lead to efficiency issues, particularly when you're trying to process data in batches on a GPU. In this article, we'll dive into the optimization process of using Hugging Face’s transformers library for a batch-processing pipeline on a GPU. I'll cover the challenges faced, the bottlenecks encountered, and the solutions we implemented to maximize efficiency.

## Problem Overview:
Fistly, we have scrapped a data from reddit on different topics and then, aimed to use Hugging Face’s zero-shot classification pipeline to classify text data on a GPU. The dataset was sizable, and running the classification sequentially proved to be too slow. Therefore, our primary goal was to implement batch processing on the GPU, leveraging its parallel computing capabilities to achieve faster inference times.

When we started using the transformers.pipeline API with .apply(), we noticed that it ran sequentially, processing each row one at a time. Additionally, we encountered the following challenges:
- Sequential Processing: Despite using .apply() on a DataFrame with GPU inference enabled, the pipeline processed each text entry sequentially, which was inefficient.
- Batch Processing Limitations: While we attempted batching, the pipeline's behavior did not fully utilize the GPU’s parallel capabilities.
- Inference Speed: As the batch size increased, we observed that inference time scaled linearly. This led us to conclude that the model was still not fully exploiting GPU resources.3

### Solution Roadmap
To address these issues, we went through a structured process of optimizing each part of our setup. Below, we outline the steps and considerations we took, which included switching to direct model inference, using mixed precision, and experimenting with batch sizes to find the most efficient configuration.

### Part 1: Data Collection Using Reddit Web Scraping
To build a model that can classify and analyze different topics effectively, we needed a robust dataset containing diverse topics. To achieve this, we decided to collect data from Reddit, a rich source of user-generated content with multiple categories and real-time discussions.

#### Step 1: Defining Topics and Data Limits
Choosing Topics:
We selected a range of 16 topics to capture a broad variety of content types. These topics included climate change, technology, sports, finance, health, education, gaming, politics, and more. By focusing on these specific topics, we aimed to create a balanced dataset that would represent various domains.

Setting Data Limits:
We decided to scrape 100 posts per topic. This limit ensured that we gathered a representative sample while keeping the data processing manageable. Adjusting the limit variable allows for scalability in case more or fewer data points are needed.

#### Step 2: Writing the Web Scraping Loop
To scrape data, we used the PRAW (Python Reddit API Wrapper) library, which provides a straightforward way to interact with the Reddit API. For each topic, we scraped the hot posts from the subreddit related to that topic, capturing the following fields:

Title: The title of the Reddit post, which provides the primary text we’ll classify.
Body: The main content or selftext of the post, providing additional context.
Subreddit: The name of the subreddit to identify where the topic is discussed.
Score: The upvote score of the post, indicating its popularity.
URL: The link to the Reddit post.
Date: The date and time when the post was created.

Here’s the loop structure that iterates over each topic and scrapes the relevant data:

for topic in topics:
    print(f"Scraping data for topic: {topic}")

    for submission in reddit.subreddit(topic).hot(limit=limit):
        posts_data.append({
            "Topic": topic,
            "Title": submission.title,
            "Body": submission.selftext,
            "Subreddit": submission.subreddit.display_name,
            "Score": submission.score,
            "URL": submission.url,
            "Date": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(submission.created_utc))
        })
#### Step 3: Overcoming Challenges in Web Scraping
While scraping data from Reddit, we encountered a few challenges:

- Rate Limiting:
Reddit enforces strict rate limits on its API usage. To avoid being blocked, we implemented a delay between requests and made sure our requests stayed within Reddit’s API usage guidelines.

- Subreddit Availability:
Some topics might not have a dedicated subreddit or may have low activity. We had to verify subreddit names and replace any missing subreddits with alternative ones.

- Incomplete or Limited Data:
Some posts do not contain a body (selftext), only a title. We handled this by considering the title as the main content and treating the body as supplementary information.

#### Step 4: Storing and Processing Data
After collecting data, we stored it in a structured format to facilitate further processing. This dataset was then cleaned and preprocessed for text classification, as outlined in earlier parts of this article.

#### Key Takeaways from the Data Collection Process
- Data Diversity:
Using multiple subreddits for different topics allowed us to build a dataset that reflects diverse discussions and perspectives.

- API Management:
Being mindful of Reddit’s rate limits and API rules is essential for successful data scraping.

- Data Quality:
Ensuring data quality by handling incomplete fields and validating topics is key to building a reliable dataset for training our classification model.

This Reddit scraping approach enabled us to gather a robust dataset quickly, creating a solid foundation for training and testing our model. In the next steps, we moved on to data preprocessing, feature engineering, and optimizing our classification pipeline.


### Part 2: Optimizing Hugging Face Transformers Pipelines for Efficient Batch Processing on GPU
#### Step 1: Initial Pipeline and Sequential Processing
Our first approach used Hugging Face’s transformers.pipeline with the zero-shot-classification model. We attempted to classify each entry in a DataFrame column by applying the pipeline to each row, assuming that .apply() would process entries in a vectorized way on the GPU.

- #### Key Insight:
Despite using GPU processing, the pipeline still handled each input in a sequential manner, due to the underlying implementation of .apply(), which wasn’t optimized for GPU batching with the pipeline API.

- #### Solution Approach:
After realizing this limitation, we began exploring the possibility of using batch processing to send multiple text inputs to the GPU simultaneously.

#### Step 2: Implementing Batch Processing
To enable batch processing, we divided the dataset into smaller subsets (batches) and processed them within a loop. This allowed us to pass multiple text inputs to the pipeline at once.

#### Challenge 1: Managing Batch Size
When we increased batch size to improve processing efficiency, the GPU became overwhelmed at a certain threshold, causing either a slowdown or out-of-memory (OOM) errors. We discovered that the pipeline's batching wasn't scaling efficiently with larger batch sizes.

#### Challenge 2: Linear Scaling
Despite using batches, processing time increased linearly with batch size, which suggested that the pipeline was handling inputs sequentially within each batch rather than taking full advantage of GPU parallelism.

- #### Solution Approach:
We realized that the standard transformers.pipeline might be adding overhead that prevented optimal batching. To gain more control, we decided to skip the pipeline altogether and directly use the model’s forward pass, which allowed us to handle batches manually.

#### Step 3: Switching to Direct Model Inference for Custom Batch Processing
To bypass the pipeline's limitations, we loaded the model and tokenizer directly and manually implemented the forward pass for each batch. This approach gave us more control over how inputs were tokenized and fed into the model, as well as how the output was processed.

#### Key Advantages of Direct Inference:

- Control Over Data Flow: By directly using the model’s forward pass, we controlled the entire workflow, from input tokenization to final classification.
- Batch Optimization: We could experiment with batch sizes more flexibly, using memory-based adjustments to maximize GPU utilization without hitting OOM errors.
- Reduced Overhead: Avoiding the pipeline reduced the latency associated with its setup and teardown for each input, leading to faster processing times.
##### Observation:
With this setup, we observed a significant improvement in speed. However, as batch size increased, we noticed that processing time initially increased logarithmically before plateauing and then scaling linearly. This pattern indicated that, at smaller batch sizes, the GPU could fully utilize parallel processing, but at larger batch sizes, memory and computational resources became the bottleneck.

#### Step 4: Fine-Tuning Batch Size for Optimal Performance
To find the "sweet spot" for batch size, we conducted experiments with varying batch sizes, noting processing time and GPU memory usage for each.

##### Key Findings:

- Logarithmic Scaling at Small Batch Sizes: For small batch sizes, processing time scaled logarithmically, as the GPU could process multiple inputs in parallel without hitting memory limits.
- Linear Scaling at Larger Batch Sizes: Beyond a certain batch size, the GPU’s memory became saturated, and the model's computation time increased in a more linear fashion, suggesting a resource limitation.
- Optimal Batch Size Range: We identified a range of batch sizes (around 10-20) that balanced efficient GPU usage with manageable memory consumption.

#### Step 5: Experimenting with Batch Sizes to Optimize Performance

we optimized our model by leveraging batch processing, but we discovered that batch size could significantly impact the performance. Therefore, we set out to conduct a systematic experiment to determine the optimal batch size that maximizes speed without overloading our system’s resources.

- Why Batch Size Matters
Batch size is a critical parameter in deep learning and machine learning workflows. Larger batches can make the process faster by allowing the model to process multiple inputs in parallel. However, they also increase memory usage, which can slow down processing or even cause the system to run out of memory. Smaller batches, on the other hand, might be inefficient because they don't take full advantage of the GPU’s parallel processing capabilities. Thus, finding the optimal batch size is essential to balancing speed and resource efficiency.

- Designing an Experiment to Test Different Batch Sizes
To empirically determine the impact of different batch sizes on processing time, we created a script to automate testing with various batch sizes. This script iterates over batch sizes in increments of 5, starting from a batch size of 5 and going up to 40. For each batch size, the script:

Extracts a sample of texts based on the current batch size.
Records the processing time for classifying a batch of that size.
Logs the results (batch size and processing time) to a dataframe for analysis.
The output of this experiment will give us a clear understanding of how batch size affects processing time, which can help in choosing an efficient batch size for the final deployment.

#### Step 6: Directly Leveraging the Model's Forward Pass for Increased Efficiency
After experimenting with batch sizes in Step 5, we turned our focus to further optimizing the inference process by using the model’s direct forward pass. This approach bypasses the usual pipeline interface in Hugging Face’s Transformers library, giving us more control and, ultimately, increased efficiency. The forward pass method is particularly effective for batch processing and helps reduce latency by avoiding the overhead associated with the pipeline method.

- Why the Forward Pass is More Efficient
Using the pipeline interface from Hugging Face is convenient, but it comes with some inherent limitations in terms of efficiency, especially when applied to large-scale batch processing. The pipeline internally manages tokenization, model inference, and label mapping, which is helpful for general use cases but adds extra computational layers. By directly using the model’s forward method, we can:

- Control Batch Processing: Custom batch management allows us to better utilize GPU resources.
- Reduce Overhead: Avoiding pipeline wrappers minimizes the computational overhead, making it more efficient, especially for repetitive tasks.
- Optimize Label Matching: By precomputing label tokens, we streamline the process of scoring and ranking candidate labels, which is particularly useful for tasks like zero-shot classification.

#### Implementing the Forward Pass for Batch Processing
Here’s how we set up the model for a more efficient forward pass:

- Model and Tokenizer Setup:
We load the AutoModelForSequenceClassification and AutoTokenizer for the facebook/bart-large-mnli model, moving both to the GPU for faster computation.

- Pre-Tokenizing Candidate Labels:
Candidate labels are tokenized and cached in advance. This pre-tokenization is stored on the GPU, allowing us to avoid redundant computations during inference, thereby enhancing efficiency.

- Batch Classification Function:
The classify_batch_direct function takes a batch of texts, tokenizes them, runs the forward pass on the model, and outputs predicted labels. Using torch.no_grad() ensures that no gradients are computed, which saves memory and speeds up inference.

 ## Conclusion
In this project, we tackled the challenges of efficiently classifying large amounts of Reddit data across various topics using a zero-shot classification model. By systematically optimizing each component of our pipeline, we achieved significant improvements in processing speed and scalability. Here’s a summary of our journey and findings:

- Data Collection: We scraped Reddit data across multiple topics, establishing a diverse dataset to apply topic classification. The data included text titles and content that we categorized using a zero-shot classifier.

- Initial Classification Pipeline: We started with a straightforward Hugging Face pipeline using facebook/bart-large-mnli for zero-shot classification. While effective, this approach quickly became computationally expensive when processing large batches of text due to its sequential nature.

- Batch Size Experimentation: Understanding the impact of batch size was a key step in our optimization. We observed that as batch sizes increased, the processing time initially grew in a nonlinear, logarithmic way, then linearly for larger batches. This experimentation revealed that batch size plays a critical role in GPU utilization and efficiency.

- Direct Model Forward Pass: To further optimize performance, we bypassed the Hugging Face pipeline and directly leveraged the model’s forward pass. By pre-tokenizing candidate labels and working directly with the logits output, we reduced the computational overhead and gained fine-grained control over batch processing. This approach significantly increased the processing speed and allowed us to handle larger data volumes more efficiently.

- Performance Gains: Through these optimizations, we achieved substantial speed improvements. Using the model’s forward pass allowed us to scale up our batch sizes while minimizing the linear slowdown, which was a limitation in the pipeline approach.

### Lessons Learned:

- Batch Processing: Optimizing batch size is crucial when working with GPU resources. Small batch sizes underutilize GPU capacity, while excessively large batches lead to linear slowdowns.
- Pipeline vs. Direct Model Access: Hugging Face pipelines are convenient but come with performance trade-offs, particularly for high-throughput tasks. Direct access to the model’s forward pass provides greater control and efficiency, making it ideal for large datasets.
- 
In conclusion, by refining our batch processing strategy, leveraging the model’s forward pass, and carefully tuning our process, we developed an efficient, scalable classification solution. These steps demonstrate the importance of understanding both the hardware limitations and the software architecture to optimize deep learning tasks effectively. This optimized approach provides a strong foundation for future classification tasks, especially when applied to large-scale, real-world datasets like those scraped from social media platforms.
