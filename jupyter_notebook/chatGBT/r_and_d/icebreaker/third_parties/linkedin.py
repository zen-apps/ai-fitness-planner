import os
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../../config/dev.env")


def scrape_linkedin_profile(linkedin_profile_url: str):
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    header_dic = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}

    response = requests.get(
        api_endpoint, params={"url": linkedin_profile_url}, headers=header_dic
    )

    # print("rsponse", response.status_code, response.json())

    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data


'''
def scrape_linkedin_profile(linkedin_profile_url: str):
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""

    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    header_dic = {"Authorization": f'Bearer {os.getenv("PROXYCURL_API_KEY")}'}

    response = requests.get(
        api_endpoint, params={"url": linkedin_profile_url}, headers=header_dic
    )
    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k
        not in [
            "people_also_viewed",
            "certifications",
            "similarly_named_profiles",
            "skills",
            "recommendations",
            "volunteer_experience",
            "projects",
            "courses",
            "publications",
            "patents",
            "test_scores",
            "languages",
            "organizations",
            "accomplishments",
            "awards",
            "education",
            "experience",
        ]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data
    # gist_response = requests.get("https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json")
    # return gist_response
'''
