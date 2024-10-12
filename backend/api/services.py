import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import torch


def generate_bookmark_description(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch page: {str(e)}")

    # Parse the page using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    page_text = soup.get_text(separator=' ', strip=True)

    # Extract the meta description
    meta_description_tag = soup.find('meta', attrs={'name': 'description'})
    
    # Convert meta description to a string (handle None case)
    if meta_description_tag and 'content' in meta_description_tag.attrs:
        meta_description = meta_description_tag['content']
    else:
        meta_description = "No meta description available"

    # Generate summary of the page
    description = generate_summary(page_text)

    return meta_description, description



def generate_summary(page_text):
    summarizer = pipeline("summarization")

    # If the text is long, you might want to split it into smaller chunks
    max_input_length = 1024  # Max tokens for many models
    # print(page_text)
    chunks = [page_text[i:i + max_input_length] for i in range(0, len(page_text), max_input_length)]


    
    
    # Summarize each chunk and combine results
    summary = []
    for chunk in chunks:
        # print(chunk)
        summary_chunk = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        # print()
        # print(summary_chunk[0]['summary_text'])
        summary.append(summary_chunk[0]['summary_text'])
    
    # Combine summaries
    final_summary = ' '.join(summary)
    return final_summary
