import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

st.set_page_config(page_title="StoryScribe", page_icon="📙")
st.header('📙 Welcome to StoryScribe, your story generator and promoter!')

load_dotenv()

openai_api_key = os.environ['OPENAI_API_KEY']

# Create a sidebar for user input
st.sidebar.title("Story teller and promoter")
st.sidebar.markdown("Please enter your details and preferences below:")

llm = ChatOpenAI(model="gpt-4o-mini")

# Ask the user for age, gender and favourite movie genre
topic = st.sidebar.text_input("What is topic?", 'A dog running on the beach')
genre = st.sidebar.text_input("What is the genre?", 'Drama')
audience = st.sidebar.text_input("What is your audience?", 'Young adult')
social = st.sidebar.text_input("What is your social?", 'Instagram')

# story generator
story_template = """You are a storyteller. Given a topic, a genre and a target audience, you generate a story.

Topic: {topic}
Genre: {genre}
Audience: {audience}
Story: This is a story about the above topic, with the above genre and for the above audience:"""
story_prompt_template = PromptTemplate(input_variables=["topic", "genre", "audience"], template=story_template)
story_chain = LLMChain(llm=llm, prompt=story_prompt_template, output_key="story")

# post generator
social_template = """You are an influencer that, given a story, generate a social media post to promote the story.
The style should reflect the type of social media used.

Story: 
{story}
Social media: {social}
Review from a New York Times play critic of the above play:"""
social_prompt_template = PromptTemplate(input_variables=["story", "social"], template=social_template)
social_chain = LLMChain(llm=llm, prompt=social_prompt_template, output_key='post')

# image generator

image_template = """Generate a detailed prompt to generate an image based on the following social media post:

Social media post:
{post}

The style of the image should be oil-painted.

"""

prompt = PromptTemplate(
    input_variables=["post"],
    template=image_template,
)
image_chain = LLMChain(llm=llm, prompt=prompt, output_key='image')

# overall chain

overall_chain = SequentialChain(input_variables=['topic', 'genre', 'audience', 'social'],
                                chains=[story_chain, social_chain, image_chain],
                                output_variables=['story', 'post', 'image'], verbose=True)

if st.button('Create your post!'):
    result = overall_chain({'topic': topic, 'genre': genre, 'audience': audience, 'social': social},
                           return_only_outputs=True)
    image_url = DallEAPIWrapper().run(result['image'][:1000])
    st.subheader('Story')
    st.write(result['story'])
    st.subheader('Social Media Post')
    st.write(result['post'])
    st.image(image_url)

"""
아래에 귀하의 세부 정보와 선호 사항을 입력하세요.
주제는 무엇인가요? Adventures of The Crazy Cat & the Chill Dog
장르는 뭐예요? SF
당신의 대상 고객은 누구인가요? Young adult
당신의 소셜은 어떤가요? Instagram

results>

이야기
제목: 갤럭틱 풋

은하수의 먼 끝자락에서, 활기찬 성운이 하늘을 물들이고 행성이 환상적인 생물들과 함께 회전하는 곳에서, 두 명의 엉뚱한 친구가 살았습니다. 미친 고양이 자주와 냉정한 개 오리온. 자주는 공허함만큼 어두운 털을 ​​가지고 있었고, 별처럼 반짝이는 은빛 반점이 여기저기에 있었습니다. 반면 오리온은 우주의 혼돈 속에서 평화를 찾는 재주가 있는 느긋한 골든 리트리버였습니다.
그들의 집은 Stardust Nomad라는 기발한 우주선이었는데, 이 우주선은 다양한 성간 폐차장에서 조립한 것으로, 신뢰할 수 있는 것보다 변덕스러운 워프 드라이브가 장착되어 있었습니다. 자주는 종종 우주선의 조종 장치를 만지작거렸고, 그녀의 거친 정신은 모험의 스릴에 이끌렸고, 오리온은 우주선의 뷰포트를 통해 쏟아지는 햇살 속에서 다음 낮잠을 꿈꾸며 쉬는 것을 더 좋아했습니다.
운명의 어느 날, 활기찬 행성 Meowthara를 공전하던 중, 자주는 우주선의 인터컴을 통해 긴급 신호가 울리는 것을 우연히 들었습니다. "도와주세요! 도와주세요! 우주 수정이 도난당했습니다! 영웅이 필요합니다!" 그 목소리는 행성에서 번성한 고양이 종족인 Felinari의 현명한 장로인 Luna의 것이었습니다. 자주의 귀가 번쩍 뜨고, 그녀는 즉시 행동에 나섰습니다.
"오리온, 우리가 그들을 도와야 해요!" 그녀는 흥분으로 눈을 반짝거리며 소리쳤다.
오리온은 느긋하게 한쪽 눈을 떴다. "정말로 그래야 하나요? 방금 햇빛을 쬐려고 했는데요."
자주는 눈을 굴렸다. "어서! 이게 우리가 영웅이 될 기회야! 게다가, 우리가 얼마나 많은 간식을 얻을 수 있을지 생각해 봐!"
마지못해 한숨을 쉬며, 오리온은 몸을 쭉 뻗고 자주를 따라가며 배를 Meowthara로 몰았다. 아래의 풍경은 캔디색 나무와 액체 다이아몬드처럼 반짝이는 강이 있는 만화경과 같은 색이었다. 그들이 착륙하자, 펠리나리 무리가 모여들었고, 그들의 털은 햇빛에 반짝였다.
루나, 장로가 그들에게 다가갔고, 그녀의 목소리는 엄숙했다. "와주셔서 감사합니다. 우주 수정은 우리 행성의 마법의 원천이며, 캡틴 퍼볼이라는 사악한 우주 해적에게 도난당했습니다. 우리가 그것을 되찾지 못하면, 메오타라는 영원히 그 힘을 잃을 것입니다."
자주의 눈은 결의로 반짝였다. "캡틴 퍼볼은 어디서 찾을 수 있을까?"
"그는 Dogstar 행성에 은신처를 가지고 있어요." 루나가 대답했다. "하지만 조심하세요. 그는 교활한 함정과 로봇 쥐 군대로 유명하거든요."
"로봇 쥐?" 오리온은 낮잠을 방해할 가능성에 대한 언급에 활기를 되찾았지만, 자주는 이미 그들의 접근을 계획하고 있었습니다.
그들은 다시 스타더스트 노매드로 뛰어들었고, 자주는 독스타로 향하는 항로를 정했다. 그들이 대기권에 진입하자 함선이 덜커덕거렸고, 오리온은 흥분과 섞인 긴장감을 느꼈다. 자주는 함선을 숨겨진 만에 착륙시켰고, 그들은 해적의 요새를 보기 위해 밖을 내다보았다. 폐금속과 빛나는 네온 불빛으로 만든 거대한 성이었다.
"좋아요, 계획은 이렇습니다." 자주가 말했다. 그녀의 꼬리는 열정적으로 꿈틀거렸다. "우리는 몰래 들어가서 코스믹 크리스털을 찾고, 퍼볼이 우리가 거기에 있다는 것을 알기 전에 빠져나가자!"
오리온은 고개를 끄덕였지만, 그는 대신 몰래 들어가서 낮잠을 자고 싶어했습니다. 그들이 성에 다가가자 갑자기 윙윙거리는 소리가 들렸습니다. 로봇 쥐 떼가 그들을 향해 달려왔고, 작은 기어가 딸각거리고 딸각거렸습니다.
"자주, 문제가 생겼을 수도 있잖아!" 오리온이 소리쳤고, 그의 냉정한 태도는 사라졌다.
자주는 민첩성을 이용해 쥐들 사이를 헤치며 앞으로 달려갔다. "내 리드를 따라와!" 그녀가 소리쳤다. 고양이의 용기에 영감을 받은 오리온은 행동에 뛰어들어 자신의 크기를 이용해 쥐들을 막았고 자주는 앞으로 달려갔다.
요새 안에서 그들은 희미하게 밝혀진 복도를 기어들어가며 레이저 트랩과 보안 카메라를 피했습니다. 마침내 그들은 캡틴 퍼볼의 반짝이는 보물들로 둘러싸인 코스믹 크리스털이 찬란한 빛으로 맥박을 치는 웅장한 방에 도착했습니다.
"잡았다!" 퍼볼의 목소리가 울려 퍼지며 그림자에서 뛰어올랐고, 그의 눈에는 장난기가 가득했다. "여기 들어와서 내 것을 가져갈 수 있다고 생각해?"
자주는 가슴을 부풀렸다. "우리는 당신이 훔친 것을 되찾기 위해 여기 있습니다!"
오리온은 그 순간의 무게를 느끼며 앞으로 나아갔다. "그리고 우리는 그것 없이 떠나지 않을 거야. Meowthara의 마법을 당신 혼자만 간직할 수는 없어."
퍼볼은 웃었지만, 로봇 마우스가 갑자기 오작동하면서 소리가 끊겼고, 마우스들은 원을 그리며 돌기 시작하며 서로 부딪혔다. 자주는 그 기회를 잡았다. "지금!"
그녀는 에너지 폭발과 함께 코스믹 크리스털을 향해 뛰어들었고, 오리온은 그녀 옆으로 뛰어들었다. 발톱을 휘두르며 번쩍이는 빛 속에서 그들은 크리스털을 움켜쥐고 방에서 달려나갔고, 퍼볼은 침착함을 되찾았다.
그들은 복도를 질주하며 함정을 피하고 간신히 잡히지 않으려고 했습니다. Cosmic Crystal을 꽉 움켜쥐고 요새에서 뛰쳐나와 Stardust Nomad로 돌아갔고, Orion의 심장은 흥분으로 두근거렸습니다.
그들이 땅에서 이륙하자 자주는 웃음을 참을 수 없었다. "우리가 해냈어, 오리온! 정말 해냈어!"
오리온은 미소를 지으며, 그의 안에서 새로운 모험의 불꽃이 타올랐다. "아시다시피, 결국 그렇게 나쁘지 않았어."
코스믹 크리스털이 Meowthara로 안전하게 돌아오자 Felinari는 기뻐했습니다. Luna는 Zazu와 Orion에게 감사를 표하며, 간식과 반짝이는 명예 훈장을 선물했습니다.
그들이 별들 사이의 집으로 돌아가는 동안, 자주는 이제 햇살 속에 편안하게 몸을 웅크리고 있는 오리온을 쳐다보았다. "어떻게 생각해, 파트너? 다음 모험을 준비했어?"
오리온은 하품을 했는데, 그의 눈은 이미 처져 있었다. "로봇 쥐가 더 이상 등장하지 않는 한..."
자주는 웃음을 터뜨리며 다음 행성으로 향하는 항로를 정했고, 그녀의 마음은 우주의 설렘과 우정의 따뜻함으로 가득 찼다. 그들이 함께라면 우주가 던지는 어떤 일이라도 맞설 수 있을 거라는 걸 알았기 때문이다.

소셜 미디어 게시물
🌌✨🚀 갤럭틱 풋: 우주의 모험이 여러분을 기다립니다! 🚀✨🌌
안녕하세요, 우주 탐험가 여러분!🐾✨ 다른 어떤 것과도 다른 별 간 여행을 떠날 준비가 되셨나요? 모험심이 가득한 야생 고양이 자주와 햇살 아래 낮잠을 자는 느긋한 강아지 오리온을 만나보세요! 두 사람은 사악한 캡틴 퍼볼로부터 마법의 코스믹 크리스털을 구하기 위한 스릴 넘치는 임무를 시작합니다! 🌟
역동적인 듀오와 함께 활기찬 행성을 탐험하고, 로봇 쥐를 피하고, 우주에서 우정의 진정한 의미를 발견하세요! 🐱❤️🐶 눈부신 영상과 따뜻한 이야기가 담긴 Galactic Paws는 모든 연령대가 즐길 수 있는 즐거운 경험입니다!
뉴욕 타임즈의 전체 리뷰를 확인하고 별까지 빠르게 이동할 준비를 하세요! ⭐️✨ 그리고 라이드를 위해 좋아하는 간식을 가져오는 것을 잊지 마세요! 🍿🌌
👉 왼쪽으로 스와이프해서 장대한 모험을 엿보세요! ➡️

#GalacticPaws #CosmicAdventure #SpaceFriends #CatAndDog #SaveTheDay #NewYorkTimes #BookReview #AdventureAwaits #FurryHeroes #ReadingIsFun #InstaBooks #Bookstagram 📚❤️🌌
"""
