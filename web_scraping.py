# Import the necessary modules
import requests
import bs4
import re
import pandas as pd

# Define a function to scrape data from a single URL
def scrape_url(url):
    # Create an empty list to store the scraped data
    data = []
    # Send a GET request to the URL and get the response
    response = requests.get(url)
    # Check if the response status code is 200 (OK)
    if response.status_code == 200:
        # Parse the response content as HTML using BeautifulSoup
        soup = bs4.BeautifulSoup(response.content, "html.parser")
        # Find all the tags that contain the data we want to scrape
        # For example, we can use "p" for paragraphs, "img" for images, "a" for links, etc.
        # You can modify this according to your needs and preferences
        tags = soup.find_all(["p", "img", "a"])
        # Loop through each tag
        for tag in tags:
            # Check if the tag is a paragraph
            if tag.name == "p":
                # Get the text content of the paragraph and strip any whitespace
                text = tag.get_text().strip()
                # Check if the text is not empty
                if text:
                    # Append the text to the data list as a tuple with the tag name
                    data.append(("p", text))
            # Check if the tag is an image
            elif tag.name == "img":
                # Get the source attribute of the image, which is the URL of the image
                src = tag.get("src")
                # Check if the source is not empty
                if src:
                    # Append the source to the data list as a tuple with the tag name
                    data.append(("img", src))
            # Check if the tag is a link
            elif tag.name == "a":
                # Get the href attribute of the link, which is the URL of the link
                href = tag.get("href")
                # Get the text content of the link and strip any whitespace
                text = tag.get_text().strip()
                # Check if the href and the text are not empty
                if href and text:
                    # Append the href and the text to the data list as a tuple with the tag name
                    data.append(("a", href, text))
    # Return the data list
    return data

# Define a function to scrape data from multiple URLs
def scrape_data(urls):
    # Create an empty list to store the scraped data from all URLs
    data = []
    # Loop through each URL
    for url in urls:
        # Scrape data from the URL using the scrape_url function
        url_data = scrape_url(url)
        # Check if the URL data is not empty
        if url_data:
            # Append the URL data to the data list
            data.extend(url_data)
    # Convert the data list to a dataframe using pandas
    data = pd.DataFrame(data)
    # Return the data dataframe
    return data
