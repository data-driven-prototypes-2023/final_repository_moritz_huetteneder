from openai import OpenAI

def strenght_weakness_agent(player_data):
    #API key was deleted for security reasons
    client = OpenAI(api_key='') # API key was deleted for security reasons
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system", 
            "content": '''Based on the attributes of a player (This includes all provided stats), please provide a detailed analysis of the player's strengths and weaknesses. Highlight areas where the player excels and areas where they could improve '''
        },
        {
            "role": "user", 
            "content": f"Here is the player data: {player_data}"
        }
    ]

    )
    
    return completion.choices[0].message.content

def potential_vs_rating(data):
    client = OpenAI(api_key='') #API key was deleted for security reasons
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system", 
            "content": '''Based on the predictet future potential of a player and his overall rating at the moment, please provide a recommandation for a scout on what to do with that player '''
        },
        {
            "role": "user", 
            "content": f"Here is the predicted potential and overall rating data: {data}"
        }
    ]

    )
    
    return completion.choices[0].message.content