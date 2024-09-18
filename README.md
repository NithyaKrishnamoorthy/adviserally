## TLDR
Visit [Adviser Ally](https://advisorally.streamlit.app/). A chatbot made for financial advisors. Advise with Confidence. 

## Background

AdvisorAlly is a state-of-the-art chatbot designed to assist financial advisors in answering queries about insurance policies. By leveraging advanced NLP models and multiple data retrieval mechanisms like SQL agents, PDF retrievers, webpage fetchers, and even search engine interfaces like DuckDuckGo, AdvisorAlly aims to simplify the complex world of insurance policies, offering insights and comparisons with unparalleled ease.

In today's digital age, where clients expect quick and accurate responses, AdvisorAlly acts as a trusty sidekick to financial advisors, ensuring they have all the information they need at their fingertips.

## Getting Started

### Prerequisites

- Python 3.9
- Create a `.streamlit/secrets.toml` file, acquire secrets from author and paste it in the `.secrets.toml` file. 

### Installation

1. Clone this repository:
   ```bash
   git clone [repository-link]
   ```

2. Navigate to the project directory:
   ```bash
   cd advisor-ally
   ```

3. Create virtual environment:
    ```bash
    python -m venv .venv

    # if using macos or linux
    .venv/bin/activate

    # if using windows
    .venv\Scripts\Activate
    ```

4. Install the required packages:
   ```bash
   
   pip install -r requirements.txt
   ```

5. For first-time run, create sqlite database from `.xlsx` files in S3.
    ```bash
    python src/create_sqlite_db.py
    ```

5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

6. Open the displayed URL in your browser to interact with AdvisorAlly.

### Deploying

This repository is linked with Streamlit Community Cloud. Visit the deployed streamlit app [here](https://advisorally.streamlit.app/)

## Future Work

- [X] enable conversational agent
- [X] test pdf question-answering retrieval
- [X] test webpages question-answering retrieval
    - [ ] fix queries not routing to webpages retrieval qa retrieval
- [X] add streamlit app
- [X] add caching feature to agents and retrievers
- [X] add one-time script to create sqlite database from excel files in s3 (for sql agent)
- [ ] add few-shot prompting 
- [ ] add more pdfs



## License

This project and its contents are proprietary to the owners. Unauthorized copying, distribution, modification, public display, or performance of this material is prohibited without the express permission of the repository owner(s). All rights reserved.
