from openai import OpenAI
import os
import subprocess
from dotenv import load_dotenv
model = "llama3:latest" # Replace with your actual model name
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import subprocess

OLLAMA_MODEL = model  # change if needed


def run_ollama_prompt(prompt: str, model: str = OLLAMA_MODEL) -> str:
    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(input=prompt)

        if process.returncode != 0:
            print(f"[OLLAMA ERROR] {stderr}")
            return "unknown"

        return stdout.strip()

    except Exception as e:
        print(f"[OLLAMA EXCEPTION] {e}")
        return "unknown"

def extract_category(text):
    prompt = f'''
    Identify the category of the news in the following text. Choose from:
    - Geopolitical News, Macroeconomic News, Commodity Supply, Commodity Demand, Commodity Price Movement
    Text: {text}
    
    Return the category as one of the options listed above. If no category applies, return "Nan".
    ***IMPORTANT: give only the category value, no extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_subcategory(text, category):
    subcategories = {
        "Geopolitical News": ["Civil unrest", "Other forms of Crisis", "Embargo", "Geo-political tension", "Trade tensions"],
        "Macroeconomic News": ["Employment", "Economy", "GDP", "Bearish technical view/outlook"],
        "Commodity Supply": ["Oversupply", "Shortage", "Supply increase", "Supply decrease"],
        "Commodity Demand": ["Demand increase", "Demand decrease"],
        "Commodity Price Movement": ["Price increase", "Price decrease", "Price movement flat", "Price target/forecast increase", "Price target/forecast decrease", "Price position"]
    }
    subcategory_options = ', '.join(subcategories.get(category, []))
    prompt = f'''
    Based on the category "{category}", identify the specific subcategory for the following text. Choose from:
    {subcategory_options}
    Text: {text}
    
    Return the subcategory or "Nan" if no match.
    ***IMPORTANT: give only the subcategory value, no extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_entity(text):
    prompt = f'''
    Identify any entities in the following text, including organizations (e.g., OPEC, European Union), individuals (e.g., analysts, political leaders), locations (e.g., Middle East, North America), and geopolitical entities (e.g., countries, regions) relevant to the context of commodity news.

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "OPEC has announced new production cuts in response to market conditions."
      Output: "OPEC"
      
    - Example 2:
      Text: "The President of the United States met with Gulf oil producers to discuss supply adjustments."
      Output: "United States, Gulf oil producers"
      
    - Example 3:
      Text: "Crude oil demand in Asia has been increasing, with China leading the surge."
      Output: "Asia, China"
      
    - Example 4:
      Text: "The International Energy Agency issued a report on global oil reserves."
      Output: "International Energy Agency"
      
    - Example 5:
      Text: "Sanctions imposed by the European Union have affected Russia’s oil exports."
      Output: "European Union, Russia"

    If no entities are found, return "Nan".

    ***IMPORTANT: Return only the entity names separated by commas, or "Nan" without any extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_primary_event(text):
    prompt = f'''
    Identify and describe the primary event in the following text. The primary event should be the main cause or the initial trigger point leading to any subsequent effects. Focus on the event that sets off a chain reaction or directly influences outcomes in the text.

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "Oil prices dropped as a result of an oversupply in the global market."
      Output: "oversupply in the global market"
      
    - Example 2:
      Text: "Sanctions imposed by the United States on the oil-exporting country led to a decrease in exports."
      Output: "sanctions imposed by the United States"
      
    - Example 3:
      Text: "Increased demand for oil from emerging markets has driven prices higher."
      Output: "increased demand for oil from emerging markets"
      
    - Example 4:
      Text: "Political instability in the region caused disruptions in oil production."
      Output: "political instability in the region"
      
    - Example 5:
      Text: "A surge in shale oil production in the U.S. has lowered global oil prices."
      Output: "surge in shale oil production in the U.S."
      
    - Example 6:
      Text: "OPEC’s decision to cut oil production was aimed at stabilizing prices."
      Output: "OPEC’s decision to cut oil production"

    If no primary event is found, return "Nan".

    ***IMPORTANT: Return only the primary event description or "Nan" with no extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_event_type(text):
    prompt = f'''
    Based on the following text, classify the event type using one of the following options:
    - Cause-movement-down-loss, Cause-movement-up-gain, Civil-unrest, Crisis, Embargo, Geopolitical-tension, Grow-strong, Movement-down-loss, Movement-flat, Movement-up-gain, Negative-sentiment, Oversupply, Position-high, Position-low, Prohibiting, Shortage, Situation-deteriorate, Slow-weak, Trade-tensions.

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "Oil prices dropped sharply due to oversupply in global markets."
      Output: "Oversupply"
      
    - Example 2:
      Text: "Sanctions imposed by the European Union caused a reduction in oil exports."
      Output: "Embargo"
      
    - Example 3:
      Text: "Growing political tensions in the Middle East affected the oil supply chain."
      Output: "Geopolitical-tension"
      
    - Example 4:
      Text: "The recent surge in oil prices was driven by increased demand in emerging markets."
      Output: "Cause-movement-up-gain"
      
    - Example 5:
      Text: "Economic conditions worsened, leading to slower growth and weak demand."
      Output: "Slow-weak"
      
    - Example 6:
      Text: "A civil unrest incident in a key oil-producing country disrupted exports."
      Output: "Civil-unrest"
      
    - Example 7:
      Text: "Reports indicate a crisis due to falling crude oil reserves."
      Output: "Crisis"
      
    - Example 8:
      Text: "Trade tensions between major economies are impacting global oil prices."
      Output: "Trade-tensions"
      
    - Example 9:
      Text: "Oil prices have remained steady amid balanced supply and demand."
      Output: "Movement-flat"

    If no clear event type is found, return "Nan".

    ***IMPORTANT: Return only the event type value or "Nan" with no extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_commodity_attributes(text):
    prompt = f'''
    Identify any commodity-related terms mentioned in the following text. Focus on specific commodities, especially those related to energy and oil markets, such as "oil," "crude oil," "Brent," "WTI," "natural gas," and similar terms.

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "Brent crude prices surged due to rising global demand."
      Output: "Brent crude"
      
    - Example 2:
      Text: "WTI oil experienced a significant drop in price amid oversupply concerns."
      Output: "WTI oil"
      
    - Example 3:
      Text: "The demand for natural gas is expected to grow in the coming months."
      Output: "natural gas"
      
    - Example 4:
      Text: "Increased production of light sweet crude has impacted the market."
      Output: "light sweet crude"
      
    - Example 5:
      Text: "Commodities such as coal and LNG have seen rising prices recently."
      Output: "coal, LNG"

    If no commodity attributes are found, return "Nan".

    ***IMPORTANT: Return only the commodity attribute values separated by commas, or "Nan" without any extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_country_attributes(text):
    prompt = f'''
    Identify any country names mentioned in the following text. Focus on recognized country names from various regions (e.g., "USA," "Japan," "Russia," "China," "Germany," "Brazil").

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "The USA and China have engaged in trade talks to stabilize markets."
      Output: "USA, China"
      
    - Example 2:
      Text: "Oil exports from Russia and Saudi Arabia have increased."
      Output: "Russia, Saudi Arabia"
      
    - Example 3:
      Text: "European countries like Germany and France are key players in the energy market."
      Output: "Germany, France"
      
    - Example 4:
      Text: "Australia and Canada are major exporters of natural resources."
      Output: "Australia, Canada"
      
    - Example 5:
      Text: "Emerging economies in countries like India and Brazil are driving demand."
      Output: "India, Brazil"

    If no country names are found, return "Nan".

    ***IMPORTANT: Return only the country names separated by commas, or "Nan" without any extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_duration_attributes(text):
    prompt = f'''
    Identify any duration or time period mentioned in the following text, such as "two years," "three weeks," "six months," etc. Focus on durations that indicate the probable effect or expected timeframe of the event.

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "The sanctions are expected to last for at least two years."
      Output: "two years"
      
    - Example 2:
      Text: "Oil production cuts are anticipated over the next six months."
      Output: "six months"
      
    - Example 3:
      Text: "The demand increase could continue for a three-week period."
      Output: "three-week"
      
    - Example 4:
      Text: "Experts predict a recovery within the next quarter."
      Output: "next quarter"
      
    - Example 5:
      Text: "A short-term drop in prices is expected over the coming months."
      Output: "coming months"

    If no duration is found, return "Nan".

    ***IMPORTANT: Return only the duration values (e.g., "two years," "six months") or "Nan" without any extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_quantity_attributes(text):
    prompt = f'''
    Identify any quantities, like volumes or weights, mentioned in the following text, such as "1.8 million gallons", "26 barrels", etc.
    Text: {text}
    
    Return only the quantities or "Nan" if none found.
    ***IMPORTANT: give only the quantity values, no extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_outcome(text):
    prompt = f'''
    Identify the outcome or effect mentioned in the following text.
    Text: {text}
    
    Return only the outcome or "Nan" if not present.
    ***IMPORTANT: give only the outcome value, no extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_financial_attributes(text):
    prompt = f'''
    Identify any financial terms mentioned in the following text that represent key financial attributes impacting the situation. Focus on terms such as "supply," "demand," "output," "production," "price," "import," "export," and similar financial attributes.

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "The increase in demand and decrease in supply have led to higher oil prices."
      Output: "demand, supply, price"
      
    - Example 2:
      Text: "Production cuts have impacted global output, while exports remain steady."
      Output: "production, output, exports"
      
    - Example 3:
      Text: "Rising imports from neighboring countries have altered the trade balance."
      Output: "imports, trade balance"
      
    - Example 4:
      Text: "Low inventory levels and restricted supply are pushing prices higher."
      Output: "inventory, supply, price"
      
    - Example 5:
      Text: "Demand for natural gas is expected to increase due to colder weather."
      Output: "demand"

    If no financial attributes are found, return "Nan".

    ***IMPORTANT: Return only the financial attribute values separated by commas, or "Nan" without any extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_forecast_attributes(text):
    prompt = f'''
    Identify any forecast-related terms in the following text that indicate projections, estimates, targets, or expected outcomes. Focus on key forecast attributes likely to be affected by events, such as price targets, demand projections, supply estimates, or growth forecasts.

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "Analysts raised their oil price target to $100 per barrel for the upcoming quarter."
      Output: "price target"
      
    - Example 2:
      Text: "The demand projection for oil has increased due to rising global consumption."
      Output: "demand projection"
      
    - Example 3:
      Text: "Supply estimates suggest a decline in production by the end of the year."
      Output: "supply estimate"
      
    - Example 4:
      Text: "There is a projected growth in oil exports from OPEC countries next year."
      Output: "export growth projection"
      
    - Example 5:
      Text: "Experts have revised their GDP forecast upwards due to stronger market conditions."
      Output: "GDP forecast"
      
    - Example 6:
      Text: "Investors are betting on a surge in commodity prices in the next quarter."
      Output: "commodity price forecast"

    If no forecast attributes are found, return "Nan".

    ***IMPORTANT: Return only the forecast attribute values (e.g., "price target," "demand projection") separated by commas, or "Nan" without any extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_group_attributes(text):
    prompt = f'''
    Identify any groups mentioned in the following text that are relevant to commodity markets, such as global producers, oil producers, hedge funds, non-OECD nations, Gulf oil producers, and similar groups involved in commodity trading, production, or regulation.

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "The recent decisions by Gulf oil producers and OPEC have influenced oil prices."
      Output: "Gulf oil producers, OPEC"
      
    - Example 2:
      Text: "Hedge funds and other institutional investors are betting on oil price increases."
      Output: "hedge funds, institutional investors"
      
    - Example 3:
      Text: "Non-OECD countries and emerging markets are seeing an increase in oil demand."
      Output: "non-OECD countries, emerging markets"
      
    - Example 4:
      Text: "The Organization of Petroleum Exporting Countries has made significant cuts in production."
      Output: "Organization of Petroleum Exporting Countries (OPEC)"
      
    - Example 5:
      Text: "Major oil producers and U.S. shale producers are reacting to new market trends."
      Output: "major oil producers, U.S. shale producers"
      
    - Example 6:
      Text: "Large investment banks and private equity funds are entering the commodity market."
      Output: "investment banks, private equity funds"
      
    - Example 7:
      Text: "Oil demand from countries outside the OECD has been steadily rising."
      Output: "countries outside the OECD"
      
    - Example 8:
      Text: "The European Union and Asian producers have begun discussions on oil imports."
      Output: "European Union, Asian producers"
      
    - Example 9:
      Text: "Middle Eastern exporters and Latin American oil producers are expanding their market share."
      Output: "Middle Eastern exporters, Latin American oil producers"
      
    - Example 10:
      Text: "Environmental groups and renewable energy coalitions are lobbying for stricter regulations on oil production."
      Output: "environmental groups, renewable energy coalitions"

    If no group attributes are found, return "Nan".

    ***IMPORTANT: Return only the group attribute values separated by commas, or "Nan" with no extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_location_attributes(text):
    prompt = f'''
    Identify any locations such as global, world, domestic, Middle East, Europe mentioned in the following text.
    Text: {text}
    
    Return only the location attributes separated by commas or "Nan" if none found.
    ***IMPORTANT: give only the location attribute values, no extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_money_attributes(text):
    prompt = f'''
    Identify any monetary values mentioned in the following text, such as "$60" or "USD 50".
    Text: {text}
    
    Return only the monetary values separated by commas or "Nan" if none found.
    ***IMPORTANT: give only the money attribute values, no extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_production_unit_attributes(text):
    prompt = f'''
    Identify any production units mentioned in the following text, such as "170,000 bpd" or "400,000 barrels per day".
    Text: {text}
    
    Return only the production units separated by commas or "Nan" if none found.
    ***IMPORTANT: give only the production unit values, no extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_state_or_province(text):
    prompt = f'''
    Identify any specific state or province mentioned in the following text, such as "Washington", "Moscow", or "Cushing".
    Text: {text}
    
    Return only the state or province or "Nan" if none found.
    ***IMPORTANT: give only the state or province value, no extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_percent_attributes(text):
    prompt = f'''
    Identify any percentage values mentioned in the following text, such as "25%" or "1.4 percent".
    Text: {text}
    
    Return only the percentage values separated by commas or "Nan" if none found.
    ***IMPORTANT: give only the percent attribute values, no extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_person_attributes(text):
    prompt = f'''
    Identify any political or notable figures mentioned in the following text, such as "Trump" or "Putin".
    Text: {text}
    
    Return only the person names separated by commas or "Nan" if none found.
    ***IMPORTANT: give only the person attribute values, no extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_price_unit_attributes(text):
    prompt = f'''
    Identify any price units mentioned in the following text, such as "$100-a-barrel" or "$40 per barrel".
    Text: {text}
    
    Return only the price units separated by commas or "Nan" if none found.
    ***IMPORTANT: give only the price unit values, no extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_event_trigger(text):
    prompt = f'''
    Identify any words or phrases in the following text that indicate the occurrence of an event related to commodity prices, geopolitical tensions, or supply-demand changes. 
    Focus on key trigger words that convey price movements (e.g., "plunged," "rose"), geopolitical tensions (e.g., "sanctions," "war"), and supply-demand adjustments (e.g., "surplus," "shortage").

    Text: "{text}"
    
    - Example 1:
      Text: "Oil prices plunged due to an oversupply in global markets."
      Output: "plunged"
      
    - Example 2:
      Text: "Sanctions imposed on Venezuela have increased market tension."
      Output: "sanctions"
      
    - Example 3:
      Text: "A forecasted surge in demand is expected next quarter."
      Output: "surge"

    If no event trigger exists, return "Nan".

    ***IMPORTANT: Return only the event trigger word or phrase, with no extra text or formatting.
    '''
    return run_ollama_prompt(model, prompt)


def extract_related_entity(text):
    prompt = f'''
    Identify any additional entities contextually tied to the primary event in the following text. Related entities could include organizations, countries, groups, or individuals involved in or affected by the primary event.

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "OPEC announced production cuts, impacting oil importers like Japan and India."
      Output: "Japan, India"
      
    - Example 2:
      Text: "Sanctions by the United States affected exports from Russia and Iran."
      Output: "Russia, Iran"
      
    - Example 3:
      Text: "The recent policy changes by the European Union influenced investments by hedge funds and oil producers."
      Output: "hedge funds, oil producers"
      
    - Example 4:
      Text: "China and South Korea are expected to increase their oil imports following market stabilization."
      Output: "China, South Korea"

    If no related entities are found, return "Nan".

    ***IMPORTANT: Return only the related entities separated by commas, or "Nan" without any extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_second_event(text):
    prompt = f'''
    Identify any consequential events that stem from the primary event in the following text. Look for events that occur as a result or reaction to the primary event.

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "OPEC's decision to cut production led to a sharp increase in oil prices."
      Output: "sharp increase in oil prices"
      
    - Example 2:
      Text: "The imposition of sanctions on exports caused a significant drop in foreign trade revenue."
      Output: "significant drop in foreign trade revenue"
      
    - Example 3:
      Text: "Higher demand for natural gas due to colder weather resulted in a surge in gas prices."
      Output: "surge in gas prices"
      
    - Example 4:
      Text: "Production cuts have led to reduced supply in the global market."
      Output: "reduced supply in the global market"

    If no second event is found, return "Nan".

    ***IMPORTANT: Return only the second event value, or "Nan" without any extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_third_event(text):
    prompt = f'''
    Identify any events that are linked to the second event in the following text. Look for events that follow as a further consequence or outcome of the second event.

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "The reduction in global supply led to a rise in oil prices, which then caused inflation in energy-dependent sectors."
      Output: "inflation in energy-dependent sectors"
      
    - Example 2:
      Text: "A surge in gas prices led to increased costs for manufacturing, resulting in higher consumer prices."
      Output: "higher consumer prices"
      
    - Example 3:
      Text: "The significant drop in foreign trade revenue caused a downturn in the local economy, which led to job cuts."
      Output: "job cuts"
      
    - Example 4:
      Text: "A rise in oil prices prompted inflation, which in turn resulted in central bank intervention."
      Output: "central bank intervention"

    If no third event is found, return "Nan".

    ***IMPORTANT: Return only the third event value, or "Nan" without any extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_outcome_attribute(text):
    prompt = f'''
    Identify any specific details quantifying or qualifying the outcome in the following text. Outcome attributes may include measurements, percentages, monetary values, or descriptive phrases indicating the scale or nature of the outcome.

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "The production cut resulted in a 10% increase in oil prices."
      Output: "10% increase in oil prices"
      
    - Example 2:
      Text: "Exports saw a decrease of 500,000 barrels per day due to sanctions."
      Output: "decrease of 500,000 barrels per day"
      
    - Example 3:
      Text: "The policy change led to a significant rise in energy costs."
      Output: "significant rise in energy costs"
      
    - Example 4:
      Text: "The company experienced a loss of $2 million in quarterly revenue."
      Output: "$2 million loss in quarterly revenue"
      
    - Example 5:
      Text: "There was a minor decline in consumer demand following the announcement."
      Output: "minor decline in consumer demand"

    If no outcome attribute is found, return "Nan".

    ***IMPORTANT: Return only the outcome attribute value, or "Nan" without any extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_temporal_attribute(text):
    prompt = f'''
    Identify any time-related aspects of the event mentioned in the following text. Temporal attributes may include specific dates, time periods, durations, or phrases indicating when an event occurred or is expected to occur.

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "The price increase is expected over the next three months."
      Output: "next three months"
      
    - Example 2:
      Text: "The sanctions will be effective starting January 2023."
      Output: "starting January 2023"
      
    - Example 3:
      Text: "Oil prices dropped sharply last week."
      Output: "last week"
      
    - Example 4:
      Text: "The production boost is anticipated by the end of the year."
      Output: "by the end of the year"
      
    - Example 5:
      Text: "Demand for crude oil is projected to rise throughout the first quarter."
      Output: "throughout the first quarter"

    If no temporal attribute is found, return "Nan".

    ***IMPORTANT: Return only the temporal attribute value, or "Nan" without any extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_polarity(text):
    prompt = f'''
    Analyze the polarity of the event described in the following text. Use the following guidelines to determine polarity:
    
    - Indicate "Positive" if the event shows favorable or growth-oriented prospects (e.g., "boost," "increase," "rise").
    - Indicate "Negative" if the event indicates adverse or declining prospects (e.g., "decline," "plunge," "drop," "sanctions").
    - Indicate "Neutral" if the event is indeterminate or stable (e.g., "unchanged," "steady").

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "Oil prices surged due to increased demand."
      Output: "Positive"
      
    - Example 2:
      Text: "Economic sanctions led to a decline in oil exports."
      Output: "Negative"
      
    - Example 3:
      Text: "Prices remained flat as supply met demand."
      Output: "Neutral"

    If no clear polarity is found, return "Nan".

    ***IMPORTANT: Return only the polarity value ("Positive," "Negative," "Neutral") or "Nan" with no extra text or formatting.
    '''
    return run_ollama_prompt(model, prompt)


def extract_modality(text):
    prompt = f'''
    Analyze the modality of the event in the following text. Choose from the following options:
    
    - Use "Asserted" if the event is a factual, real occurrence (e.g., "oil prices rose," "sanctions imposed").
    - Use "Believed Event" if the event is speculative or based on opinions (e.g., "expected increase," "analysts predict").
    - Use "Hypothetical Event" if the event is hypothetical or uncertain (e.g., "could happen," "might lead to").

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "Oil prices rose sharply after sanctions were imposed."
      Output: "Asserted"
      
    - Example 2:
      Text: "Analysts predict a potential increase in oil demand."
      Output: "Believed Event"
      
    - Example 3:
      Text: "If production cuts continue, prices might stabilize."
      Output: "Hypothetical Event"

    If no clear modality is found, return "Nan".

    ***IMPORTANT: Return only the modality value ("Asserted," "Believed Event," or "Hypothetical Event") or "Nan" without any extra text.
    '''
    return run_ollama_prompt(model, prompt)


def extract_causal_links(text):
    prompt = f'''
    Identify any causal relationships between events in the following text. Focus on phrases or wording that indicate one event directly causes or influences another.
    
    Use these guidelines:
    - Explicit causal links may be indicated by words like "because," "due to," "as a result of," "caused by," or "leads to."
    - Implicit causality may be inferred from context where one event logically results from another, even if explicit causal words are absent.

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "Oil prices dropped due to an oversupply in the global market."
      Output: "prices dropped due to an oversupply"
      
    - Example 2:
      Text: "The sanctions imposed on the country led to a reduction in oil exports."
      Output: "sanctions led to a reduction in exports"
      
    - Example 3:
      Text: "As supply levels decreased, oil prices surged to new highs."
      Output: "supply levels decreased, causing prices to surge"
      
    - Example 4 (Implicit Causality):
      Text: "Oil production was reduced, pushing prices higher amid rising demand."
      Output: "production was reduced, pushing prices higher"

    If no causal links are found, return "Nan".

    ***IMPORTANT: Return only the causal link phrase (e.g., "X due to Y") or "Nan" with no extra text or formatting.
    '''
    return run_ollama_prompt(model, prompt)


def extract_influencing_factors(text):
    prompt = f'''
    Identify any factors that significantly influence the outcomes of the events in the following text. Common influencing factors in commodity news include supply, demand, geopolitical tensions, economic policies, sanctions, and market sentiment. Focus on words or phrases that represent these or similar factors.

    Text: "{text}"

    Examples:
    - Example 1:
      Text: "The increase in demand and a shortage in supply led to higher oil prices."
      Output: "demand, supply"
      
    - Example 2:
      Text: "Economic sanctions and political instability in the region impacted the market."
      Output: "sanctions, political instability"
      
    - Example 3:
      Text: "A surge in demand and favorable weather conditions boosted agricultural exports."
      Output: "demand, weather conditions"
      
    - Example 4:
      Text: "Fears of a recession have created negative sentiment across markets."
      Output: "recession fears, market sentiment"

    If no influencing factors are found, return "Nan".

    ***IMPORTANT: Return only the influencing factors separated by commas, or "Nan" with no extra text.
    '''
    return run_ollama_prompt(model, prompt)




# ---------------------------------------------------------------------------
# extract_oil_summary - Chain-of-Thought oil-focused summariser
#
# Used to populate Brahmanda's `cleaned_text` field AND as the input text for
# the BERT embedding (so cosine similarity is anchored to oil-market content,
# not Factiva boilerplate or off-topic paragraphs).
#
# The prompt walks the model through 4 explicit reasoning steps (relevance,
# key drivers, causal factors, final summary) and asks it to return the
# answer in a labelled block we can parse deterministically. If the article
# is not oil-related, the function returns the literal string
# "NOT_OIL_RELEVANT" so the caller can decide what to embed instead.
# ---------------------------------------------------------------------------
import re as _re_oil  # local alias to avoid clobbering any module-level `re`

_OIL_SUMMARY_COT_PROMPT = '''You are an oil-market analyst. Read the article below and produce an OIL-MARKET-RELEVANT summary by reasoning step by step.

Follow this exact chain of thought:

STEP 1 - Relevance check.
Does the article discuss crude oil, refined products, OPEC/OPEC+, oil-producing countries, oil supply or demand, oil inventories, oil prices, refinery activity, oil-related geopolitics, or oil-market forecasts?
- If NO, stop and return exactly: NOT_OIL_RELEVANT
- If YES, continue.

STEP 2 - Identify the key oil-market drivers mentioned.
List the supply-side, demand-side, geopolitical, macroeconomic, inventory, OPEC/production, refining, and forecast signals you find. Drop sports, weather (unless it moves oil), unrelated business, opinion that does not move markets.

STEP 3 - Identify causal factors.
For each driver, note WHY it matters for oil prices (e.g., "OPEC+ output cut -> tighter supply -> bullish for prices").

STEP 4 - Compose the final summary.
Write a clean, factual 4-6 sentence summary covering ONLY the oil-relevant material. Include the causal factors from step 3 in the prose. No bullets, no markdown, no preamble.

Article:
"""
{text}
"""

Return your answer in EXACTLY this format (and nothing else):

<reasoning>
Step 1: ...
Step 2: ...
Step 3: ...
</reasoning>
<summary>
... your 4-6 sentence summary here, OR the literal string NOT_OIL_RELEVANT ...
</summary>
'''


def _parse_oil_summary(raw: str) -> str:
    """Pull the <summary>...</summary> block out of the CoT response.
    Falls back gracefully if the model did not emit the tags."""
    if not raw:
        return ""
    m = _re_oil.search(r"<summary>\s*(.*?)\s*</summary>", raw, _re_oil.DOTALL | _re_oil.IGNORECASE)
    if m:
        out = m.group(1).strip()
    else:
        out = _re_oil.sub(r"<reasoning>.*?</reasoning>", "", raw,
                          flags=_re_oil.DOTALL | _re_oil.IGNORECASE).strip()
    if "NOT_OIL_RELEVANT" in out.upper():
        return "NOT_OIL_RELEVANT"
    return out


def extract_oil_summary(text: str, max_chars: int = 8000) -> str:
    """Chain-of-Thought oil-market summariser.

    Returns
    -------
    A 4-6 sentence summary of the article's oil-market content, OR the literal
    string "NOT_OIL_RELEVANT" if the article is not about oil. Returns empty
    string if `text` is empty/None.
    """
    if not text or not str(text).strip():
        return ""
    prompt = _OIL_SUMMARY_COT_PROMPT.format(text=str(text)[:max_chars])
    raw = run_ollama_prompt(model, prompt)
    return _parse_oil_summary(raw)
