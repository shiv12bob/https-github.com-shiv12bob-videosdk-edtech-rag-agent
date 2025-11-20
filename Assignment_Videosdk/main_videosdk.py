#!/usr/bin/env python3
import os, asyncio
from dotenv import load_dotenv
load_dotenv()
from rag import ask_with_rag

# VideoSDK imports may require installation of videosdk-agents and provider extras
try:
    from videosdk.agents import Agent, AgentSession, CascadingPipeline, JobContext, RoomOptions, WorkerJob, ConversationFlow
    from videosdk.plugins.silero import SileroVAD
    from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
    from videosdk.plugins.deepgram import DeepgramSTT
    from videosdk.plugins.openai import OpenAILLM
    from videosdk.plugins.elevenlabs import ElevenLabsTTS
except Exception as e:
    print("VideoSDK imports failed (you may not have videosdk-agents installed):", e)
    raise

try:
    pre_download_model()
except Exception:
    pass

class RagAgent(Agent):
    def __init__(self, instructions: str = None):
        instructions = instructions or (
            "You are an AI assistant specialized in **EdTech sales**.\n"
            "- Always think like an EdTech sales consultant.\n"
            "- Use the provided company/product documentation and playbooks first.\n"
            "- Focus on: lead qualification, demo pitching, objection handling, pricing discussion, and upsell/cross-sell for EdTech products.\n"
            "- If the answer is not in the docs, you may use your own knowledge, but keep it framed around EdTech sales."
        )
        super().__init__(instructions=instructions)

    async def on_enter(self):
        await self.session.say("Hello! I am your RAG-enabled assistant. Ask me anything.")

    async def on_exit(self):
        await self.session.say("Goodbye!")

    async def on_user_message(self, user_text: str):
        try:
            answer = ask_with_rag(user_text)
        except Exception as e:
            answer = f"Error while retrieving answer: {e}"
        await self.session.say(answer)

async def start_cascading(context: JobContext):
    agent = RagAgent()
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt=DeepgramSTT(model="nova", language="en"),
        llm=OpenAILLM(model=os.getenv("OPENAI_MODEL", "gpt-4o")),
        tts=ElevenLabsTTS(voice="alloy"),
        vad=SileroVAD(threshold=0.35),
        turn_detector=TurnDetector(threshold=0.8)
    )

    session = AgentSession(agent=agent, pipeline=pipeline, conversation_flow=conversation_flow)
    try:
        await context.connect()
        await session.start()
        await asyncio.Event().wait()
    finally:
        await session.close()
        await context.shutdown()

def make_context():
    return JobContext(room_options=RoomOptions(room_id=os.getenv("VIDEOSDK_ROOM_ID", None), name="VideoSDK RAG Agent", playground=True))

if __name__ == "__main__":
    mode = os.getenv("VIDEOSDK_PIPELINE", "cascading").lower()
    job = None
    if mode == "realtime":
        job = WorkerJob(entrypoint=start_cascading,jobctx=make_context())
    else:
        job = WorkerJob(entrypoint=start_cascading,jobctx=make_context())
    print("Starting VideoSDK Agent job (mode=%s)..." % mode)
    job.start()
