from typing import Tuple
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from chains.custom_chains import (
    get_summary_chain,
    get_interests_chain,
    get_ice_breaker_chain,
)
from third_parties.linkedin import scrape_linkedin_profile

from output_parsers import (
    summary_parser,
    topics_of_interest_parser,
    ice_breaker_parser,
    Summary,
    IceBreaker,
    TopicOfInterest,
)

name = "Harrison Chase"
linkedin_username = linkedin_lookup_agent(name=name)
linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)

summary_chain = get_summary_chain()
summary_and_facts = summary_chain.run(information=linkedin_data)
summary_and_facts = summary_parser.parse(summary_and_facts)

interests_chain = get_interests_chain()
interests = interests_chain.run(information=linkedin_data)
interests = topics_of_interest_parser.parse(interests)

ice_breaker_chain = get_ice_breaker_chain()
ice_breakers = ice_breaker_chain.run(information=linkedin_data)
ice_breakers = ice_breaker_parser.parse(ice_breakers)

print(
    summary_and_facts,
    interests,
    ice_breakers,
    linkedin_data.get("profile_pic_url"),
)
