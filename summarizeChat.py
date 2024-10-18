import asyncio
import sys
import uvloop
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from time import sleep
import emoji
import grapheme
from groq import Groq
import pytz
import os
import ollama
from transformers import GPT2Tokenizer
from pyrogram import Client
from pyrogram.enums.parse_mode import ParseMode
from pyrogram.types.messages_and_media.message_reactions import MessageReactions
from pyrogram.types.messages_and_media.reaction import Reaction
from pyrogram.types.messages_and_media.message import Message
from pyrogram.types.messages_and_media.message_entity import MessageEntity
from pyrogram.enums.message_entity_type import MessageEntityType
from pyrogram.enums.chat_type import ChatType
from pyrogram.types import Chat

import yaml
import re
from dataclasses import dataclass, field
from typing import List, Tuple

uvloop.install()

api_key = os.environ.get("GROQ_API_KEY")


@dataclass
class ChannelConfig:
    id: int
    name: str
    filters: list[str] = field(default_factory=list)
    language: str = None
    context: str = None


# Load configuration from a YAML file
def load_config(config_file="config.yaml"):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


# Initialize the tokenizer globally
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


format_style = """Format using emojis, bbbullet points, and other visual elements to make the summary engaging and easy to read."""


def is_private_group(chat: Chat) -> bool:
    try:
        return chat.type in ["group", "supergroup"] and not chat.username
    except Exception as e:
        print(f"Error checking chat: {e}")
        return False


def make_hashtag(text: str) -> str:
    return "#" + text.lower().replace(" ", "").replace(".", "").replace(
        "-", ""
    ).replace("/", "").replace("(", "").replace(")", "").replace(",", "").replace("|", "")


def pick_unicore_emoji(name: str) -> str:
    em = emoji.EMOJI_ALIAS_UNICODE[":headphone:"]
    return em


def count_offsets(text: str) -> int:
    # be aware of the new line character and emojis and non-ascii characters
    return len(text.encode("utf-8")) + text.count("\n") + text.count(emoji.emojize(":"))


def format_user_friendly_date(date: datetime) -> str:
    return date.strftime("%A, %B %d, %Y")


def create_collapsible_quote(*lines, hidden=None):
    """
    Create a collapsible quote for Telegram messages using Markdown syntax.

    Args:
    *lines: Variable number of strings, each representing a visible line in the quote.
    hidden: Optional string or list of strings to be hidden (expandable part).

    Returns:
    str: Markdown formatted string for a collapsible quote.
    """
    quote = ">" + "\n>".join(lines)

    if hidden:
        if isinstance(hidden, list):
            hidden_text = "\n>".join(hidden)
        else:
            hidden_text = hidden
        quote += f"\n>{hidden_text}||"

    return quote


class SummarizationModel:
    def __init__(self):
        self.context = ""

    def chat(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses should implement this method")

    def set_context(self, context: str):
        self.context = context


class OllamaModel(SummarizationModel):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def chat(self, prompt: str) -> str:
        if self.context:
            prompt = f"{prompt}\n\n{self.context}"

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return response["message"]["content"].strip()


class GroqModel(SummarizationModel):
    def __init__(self, model_name: str):
        super().__init__()
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def chat(self, prompt: str) -> str:
        if self.context:
            prompt = f"{prompt}\n\n{self.context}"
        sleep(1)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            # temperature=1,
            # max_tokens=1024,
            # top_p=1,
            stream=True,
            # stop=None,
        )

        response = ""
        for chunk in completion:
            if isinstance(chunk, tuple):
                # rate limit exceeded
                sleep(1)
                continue
            c = chunk.choices
            chunk_choice = c[0]
            response += chunk_choice.delta.content or ""
        return response.strip()


@dataclass
class MessageInfo:
    id: int
    text: str
    author_id: int | None
    author_name: str | None
    reactions: int
    url: str | None = None
    replies: List["MessageInfo"] = field(default_factory=list)
    reply_to_message_id: int | None = None

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    # lt
    def __lt__(self, other):
        return self.id < other.id

    @staticmethod
    def print_conversation(message: "MessageInfo", indent: int = 0):
        print(f"{'  ' * indent}{message.author_name}: {message.text}")
        for reply in message.replies:
            MessageInfo.print_conversation(reply, indent + 1)


@dataclass
class ThreadInfo:
    root_message: MessageInfo
    depth: int
    total_reactions: int
    unique_participants: set[int]

    def __str__(self):
        url = self.root_message.url or "NA"
        author = self.root_message.author_name or "NA"
        replies = len(self.root_message.replies)
        return f"🧵 Thread: {url} by {author} with {replies} replies and {self.total_reactions} reactions 👍"


@dataclass
class MsgAnalysis:
    reaction_total_count: int
    unique_reactions: int
    top_reactions: List[Tuple[str, int]]
    message: Message
    replies_count: int = 0
    msg_link: str = None
    hour_of_day: int = None
    user_course: str = None
    is_admin: bool = False


def get_chat_name(message: Message) -> str:
    return message.chat.title or message.chat.username or message.chat.first_name


class TelegramSummarizer:
    def reload_config(self):
        self.config = load_config()
        config = self.config
        self.channels = config["channels"]
        default_model_provider = config.get("model_provider", "ollama")
        default_model_name = config.get("model_name", "llama3.2")
        if default_model_provider == "ollama":
            self.model = OllamaModel(default_model_name)
        elif default_model_provider == "groq":
            self.model = GroqModel(default_model_name)
        else:
            raise ValueError("Invalid model provider")
        self.max_length = config.get("max_length", 4000)
        self.summarization_frequency_hours = config.get("summarization_frequency", 24)
        self.output_dir = config.get("output_dir", "summaries")
        self.attachments_dir = os.path.join(self.output_dir, "attachments")
        self.urls_dir = os.path.join(self.output_dir, "urls")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.attachments_dir, exist_ok=True)
        os.makedirs(self.urls_dir, exist_ok=True)

    def __init__(self):
        self.reload_config()
        config = self.config
        self.app = Client(
            config["session_name"], api_id=config["api_id"], api_hash=config["api_hash"]
        )
        self.thread_cache: dict[str, ThreadInfo] = {}
        self.reload_config()

        # Regex pattern for URLs
        self.url_pattern = re.compile(r"(https?://\S+)")

    async def fetch_dialogs(self):
        # Fetch and print available dialogs (channels and chats)
        async with self.app:
            tex = """
            
"""
            await self.app.send_message(chat_id="me", text=tex)
            dialogs = self.app.get_dialogs()
            chat_map = {}
            async for dialog in dialogs:
                title = dialog.chat.first_name or dialog.chat.title
                chat_id = dialog.chat.id
                chat_map[title] = chat_id
            return chat_map

    async def process_channels(self):
        # Calculate the time offset based on summarization frequency
        async with self.app:
            while True:
                # await self.test_formating()
                time_offset = datetime.now(pytz.utc) - timedelta(
                    hours=self.summarization_frequency_hours
                )
                for channel_info in self.channels:
                    channel_config = ChannelConfig(
                        id=channel_info["id"],
                        name=channel_info["name"],
                        filters=channel_info.get("filters", []),
                    )
                    print(f"Processing {channel_config.name}...")
                    try:
                        await self.process_chat_history(time_offset, channel_config)
                    except Exception as e:
                        print(f"Error processing {channel_config.name}: {e}")
                print("Waiting for the next cycle...")
                await asyncio.sleep(
                    self.summarization_frequency_hours * 3600
                )  # Convert hours to seconds
                self.reload_config()

    async def test_formating(self):
        tex = f"""
📊 Daily Summary: 
"""
        # entities: list[MessageEntity] = [
        #     self.create_blockquote_entity(),
        # ]
        await self.app.send_message(
            chat_id="me",
            text=tex,  # entities=entities, parse_mode=ParseMode.MARKDOWN
        )
        # https://docs.pyrogram.org/topics/text-formatting
        # await self.app.send_message(
        #     "me",
        #     "text user mention",
        #     entities=[
        #         MessageEntity(type="mention", offset=0, length=15, user=123456789)
        #     ],
        # )
        sys.exit(0)

    def create_blockquote_entity(self, offset: int, length: int) -> MessageEntity:
        return MessageEntity(
            type=MessageEntityType.BLOCKQUOTE,
            offset=offset,
            length=length,
            expandable=True,
        )

    @staticmethod
    def generate_hourly_message_summary(msg_per_hour_of_day: dict):
        blocks = "▁▂▃▄▅▆▇█"

        def normalize(value, min_value, max_value, new_min, new_max):
            if value == 0:
                return 0
            if max_value == min_value:
                return new_min
            return int(
                ((value - min_value) / (max_value - min_value)) * (new_max - new_min)
                + new_min
            )

        max_count = max(msg_per_hour_of_day.values())
        min_count = min(msg_per_hour_of_day.values())

        periods = {
            "🌅 Morning (6AM-11AM)  ": range(6, 12),
            "☀️ Afternoon (12PM-5PM)": range(12, 18),
            "🌆 Evening (6PM-11PM)  ": range(18, 24),
            "🌙 Night (12AM-5AM)    ": range(0, 6),
        }

        visualization = ""

        for period, hours in periods.items():
            visualization += f"\n{period} "
            for hour in hours:
                count = msg_per_hour_of_day.get(hour, 0)
                normalized = normalize(count, min_count, max_count, 0, len(blocks) - 1)
                visualization += blocks[normalized]

        return visualization

    def create_message_info(self, message: Message) -> MessageInfo:
        if message.empty:
            return MessageInfo(
                id=message.id,
                text="Empty message",
                author_id=None,
                author_name="Unknown",
                reactions=0,
            )
        desc = message.text
        if not desc:
            # try get media caption
            if message.media:
                desc = message.caption
            else:
                desc = "No text content"

        return MessageInfo(
            id=message.id,
            text=desc,
            author_id=message.from_user.id if message.from_user else None,
            author_name=self.get_user_str(message),
            reactions=sum(
                r.count
                for r in (message.reactions.reactions if message.reactions else [])
            ),
            url=message.link if message.link else None,
            reply_to_message_id=message.reply_to_message_id,
        )

    async def analyze_thread(self, message: Message) -> ThreadInfo:
        if message.id in self.thread_cache:
            return self.thread_cache[message.id]

        depth = 0
        total_reactions = 0
        unique_participants = set()
        message_stack = []

        current_message = message
        while True:
            message_info = self.create_message_info(current_message)
            message_stack.append(message_info)

            total_reactions += message_info.reactions
            if message_info.author_id:
                unique_participants.add(message_info.author_id)

            if not current_message.reply_to_message_id:
                break

            depth += 1
            current_message = await self.load_previous_message(
                message.chat.id,
                current_message.reply_to_message_id,
            )

        # Reconstruct the thread structure
        root_message = message_stack.pop()
        while message_stack:
            reply = message_stack.pop()
            root_message.replies.append(reply)

        thread_info = ThreadInfo(
            root_message, depth, total_reactions, unique_participants
        )
        self.thread_cache[message.id] = thread_info
        return thread_info

    async def load_previous_message(self, chat_id, message_id):
        try:
            return await self.app.get_messages(
                chat_id=chat_id,
                message_ids=message_id,
            )
        except Exception as e:
            print(f"Error fetching message {message_id}: {e}")
            return None

    async def analyze_message(
        self, message, conversation_replies: dict, thread_roots: dict[int, ThreadInfo]
    ) -> MsgAnalysis | None:
        if not message:
            return None

        # Extract reactions
        reaction_total_count = 0
        unique_reactions = 0
        top_reactions = []
        if message.reactions:
            reaction_list: list[Reaction] = message.reactions.reactions
            reaction_total_count = sum(reaction.count for reaction in reaction_list)
            reaction_types = Counter(reaction.emoji for reaction in reaction_list)
            unique_reactions = len(reaction_types)
            top_reactions = reaction_types.most_common(3)

        "https://t.me/investinnl/152719"

        reply_to_top_message_id = message.reply_to_top_message_id
        reply_to_message_id = message.reply_to_message_id
        if reply_to_message_id:
            print(f"Reply to message: {message.text}")
            # conversation_replies[reply_to_message_id].append(message.id)
            thread_info = await self.analyze_thread(message)
            root_id = thread_info.root_message.id

            if (
                root_id not in thread_roots
                or thread_roots[root_id].depth < thread_info.depth
            ):
                thread_roots[root_id] = thread_info

        if reply_to_top_message_id:
            print(f"Reply to top message: {message.text}")
            conversation_replies[reply_to_top_message_id].append(message.id)

        if_foward_from = message.forward_from
        if if_foward_from:
            print(f"Forwarded from: {message.text}")

        # Extract replies count
        # replies_count = message.replies.replies if message.replies else 0
        # if replies_count:
        #     print(f"Replies count: {replies_count}")

        # Extract message landing page
        msg_link = message.link if message.link else ""

        # Extract hour of the day
        hour_of_day = message.date.hour

        # Extract user course (assuming user course is a custom field in the message)
        user_course = (
            message.from_user.course
            if message.from_user and hasattr(message.from_user, "course")
            else ""
        )

        is_amdmin = message.from_user and hasattr(message.from_user, "is_admin")
        if is_amdmin:
            print(f"Admin message: {message.text}")

        has_mentions = message.entities and any(
            entity.type == "mention" for entity in message.entities
        )
        if has_mentions:
            print(f"Mentioned user: {message.text}")

        has_hashtags = message.entities and any(
            entity.type == "hashtag" for entity in message.entities
        )
        if has_hashtags:
            print(f"Hashtags: {message.text}")

        has_media = message.media
        if has_media:
            print(f"Media: {message.text}")

        msg_len = len(message.text) if message.text else 0

        if hasattr(message, "pinned"):
            is_pinned = message.pinned
            if is_pinned:
                print(f"Pinned message: {message.text}")

        # Create a message analysis object
        message_analysis = MsgAnalysis(
            reaction_total_count=reaction_total_count,
            unique_reactions=unique_reactions,
            top_reactions=top_reactions,
            # replies_count=replies_count,
            msg_link=msg_link,
            hour_of_day=hour_of_day,
            user_course=user_course,
            message=message,
            is_admin=is_amdmin,
        )

        return message_analysis

    @staticmethod
    def remove_intermediate_roots(thread_roots: dict[int, ThreadInfo]):
        # Remove intermediate roots that are part of a longer thread
        roots_to_remove = set()

        for root_id, thread_info in thread_roots.items():
            current_message: MessageInfo = thread_info.root_message

            for reply in current_message.replies:
                if reply.id in thread_roots:
                    roots_to_remove.add(reply.id)

        # Remove intermediate roots
        for root_id in roots_to_remove:
            del thread_roots[root_id]

    @staticmethod
    def prepare_thread_for_llm(
        thread: ThreadInfo, channel_name: str, context_info: str
    ) -> str:
        def format_message(message: MessageInfo, indent: int = 0) -> str:
            author = message.author_name or "Unknown User"
            reactions = f" [{message.reactions} reactions]" if message.reactions else ""
            formatted = f"{'  ' * indent}{author}: {message.text}{reactions}\n"
            for reply in message.replies:
                formatted += format_message(reply, indent + 1)
            return formatted

        formatted_conversation = format_message(thread.root_message)
        word_count = len(formatted_conversation.split())

        # Calculate desired summary length
        base_length = 50  # Minimum summary length
        max_length = 300  # Maximum summary length
        length_factor = min(word_count / 100, 1)  # Scale factor based on input size

        # Determine the relative size of the thread
        if word_count < 50:
            thread_size = "very short"
            summary_length = "very brief"
        elif word_count < 200:
            thread_size = "short"
            summary_length = "very brief"
        elif word_count < 500:
            thread_size = "medium-length"
            summary_length = "very brief"
        elif word_count < 1000:
            thread_size = "long"
            summary_length = "very brief"
        else:
            thread_size = "very long"
            summary_length = "very brief"

        # Adjust instructions based on thread size
        if thread_size == "very short":
            summary_instruction = f"Provide a {summary_length} summary of this {thread_size} thread. Focus on the main point or question."
        elif thread_size == "short":
            summary_instruction = f"Summarize this {thread_size} thread {summary_length}ly, highlighting the key points and any conclusions reached."
        else:
            summary_instruction = f"Provide a {summary_length} summary of this {thread_size} thread. Include the main topics discussed, key questions and answers, and overall sentiment or conclusions."

        prompt = f"""{summary_instruction}. The conversation is from the channel "{channel_name}.  {format_style}". 
    Conversation:
    {formatted_conversation}
    __
    """
        return prompt

    # Update the summarize_thread method in your class
    def summarize_thread(
        self, thread_root: ThreadInfo, channel_name: str, context_info: str
    ) -> str:
        prompt = TelegramSummarizer.prepare_thread_for_llm(
            thread_root, channel_name, context_info
        )
        try:
            return self.model.chat(prompt)
        except Exception as e:
            print(f"Error summarizing thread: {e}")
            return None

    def scrape_links(self, links: list[str]) -> dict[str, str]:
        # Scrape the content of the links and summarize them

        scraped_links = {}
        for link in links:
            scraped_links[link] = "Summary of the content"
        return scraped_links

    async def process_chat_history(self, offset, channel_config: ChannelConfig):
        msgs = []
        collected_links = []
        extracted_attachments = []
        first_url = None
        last_url = None
        entities: list[MessageEntity] = []

        active_participants = defaultdict(list)
        total_number_of_messages = 0
        msg_per_hour_of_day = defaultdict(int)
        analysed_msgs: list[MsgAnalysis] = list()
        conversation_replies: dict = defaultdict(list)
        thread_roots: dict[int, ThreadInfo] = defaultdict(ThreadInfo)

        chat_context = channel_config.context
        if channel_config.language:
            lang = channel_config.language
            # Provide insturciton to use the given language for both input and output.
            chat_context += f"Use {lang} language for both input and output."
        if chat_context:
            self.model.set_context(chat_context)

        async for message in self.app.get_chat_history(
            channel_config.id, limit=3000
        ):

            date = message.date.astimezone(pytz.utc)

            channel_name = get_chat_name(message)

            if date < offset:
                break

            total_number_of_messages += 1

            if analysis := await self.analyze_message(
                message, conversation_replies, thread_roots
            ):
                analysed_msgs.append(analysis)

            if message.from_user:
                active_participants[message.from_user.id].append(message)

            if not is_private_group(message.chat):
                if not last_url:
                    last_url = message.link
                first_url = message.link

            msg_per_hour_of_day[date.hour] += 1

            if not self.apply_filters(message, channel_config.filters):
                continue

            if links := self.extract_links(message.text):
                collected_links.extend(links)

            if attachments := await self.extract_attachments(
                message, channel_name, date
            ):
                extracted_attachments.extend(attachments)

            user = self.get_user_str(message)
            text_ = f"{user}: {message.text}"
            msgs.append(text_)

        # reverse to maintain order
        msgs.reverse()
        extracted_attachments.reverse()
        collected_links.reverse()
        links_summary = self.scrape_links(collected_links)

        if not total_number_of_messages:
            print(
                f"No messages to summarize for {channel_name} in the given time frame."
            )
            return

        # for msg_id, replies in conversation_replies.items():
        #     conversation_analysis = self.analyze_conversation(msg_id, replies)
        # Calculate importance scores and store messages with scores
        scored_messages = []
        for im in analysed_msgs:
            importance_score = im.reaction_total_count * 2 + im.unique_reactions * 3
            scored_messages.append(
                {
                    "id": im.message.id,
                    "text": im.message.text,
                    "link": im.msg_link,
                    "date": im.message.date,
                    "importance_score": importance_score,
                }
            )

        # Sort messages by importance score in descending order
        scored_messages.sort(key=lambda x: x["importance_score"], reverse=True)

        limit_msgs = min(3, len(scored_messages))
        # Select top 10 messages
        top_x_messages = scored_messages[:limit_msgs]

        # Print top 10 messages
        print(f"Top {limit_msgs} important messages in {channel_name}:")
        for msg in top_x_messages:
            print(
                f"Message ID: {msg['id']}, Score: {msg['importance_score']}, Text: {msg['text']}"
            )

        summary_text = self.generate_summaries(msgs, channel_name, offset)

        user_friendly_date = format_user_friendly_date(offset)
        highlights = f"📝 {user_friendly_date}\n"

        highlights += f" {summary_text}\n\n"

        # vizualy separate from the rest of the text
        highlights += "▔" * 5 + "\n"

        thread_limit = min(3, len(thread_roots))
        if thread_roots:
            highlights += f"\n📜 Top {thread_limit} important threads\n"
            # get currenti offset in highlights text
            magic_offset = 3
            text_offset = grapheme.length(highlights) + magic_offset
            if ti := self.add_threads_info(
                channel_name, thread_roots, limit=thread_limit
            ):
                highlights += ti

            entities.append(
                self.create_blockquote_entity(
                    text_offset,
                    grapheme.length(highlights) + magic_offset - text_offset,
                )
            )

        # vizualy separate from the rest of the text
        highlights += "\n" + "▔" * 5 + "\n"

        user_limit = min(5, len(active_participants))
        highlights += (
            f"\n👥 Active Participants from total {len(active_participants)} generated {len(msgs)} messages\n"
        )
        sorted_participants = sorted(
            active_participants.items(), key=lambda item: len(item[1]), reverse=True
        )[:user_limit]
        for idx, (user_id, messages) in enumerate(sorted_participants, 1):
            user = self.get_user_str(messages[0])
            highlights += f"   {idx}. {user} with 📨 messages: {len(messages)}\n"

        print(f"Processing all chunks for {channel_name}...")

        self.save_urls(collected_links, channel_name, offset)
        self.save_attachments_info(extracted_attachments, channel_name, offset)

        # add urls and attachments to final summary
        if first_url or last_url or collected_links:
            highlights += "\n🔗 Links:"
            if first_url:
                highlights += f"\n   • First post: {first_url}"
            if last_url:
                highlights += f"\n   • Last post:  {last_url}\n"
            for url in collected_links:
                highlights += f"   • {url}\n"

        highlights += "\n⏰ Message Frequency:"
        highlights += self.generate_hourly_message_summary(msg_per_hour_of_day) + "\n"

        highlights += f"\n\n🏷️ Tags: #summary {make_hashtag(channel_name)} {make_hashtag(self.model.model_name)}\n"

        await self.send_summary_to_channel(highlights, entities=entities)
        print(f"Summary for {channel_name} sent to channel.")

    def add_threads_info(
        self, channel_name, thread_roots: dict[int, ThreadInfo], limit: int
    ) -> str:
        self.remove_intermediate_roots(thread_roots)  # most likely not needed

        for thread_id, thread_root in thread_roots.items():
            print(f"Thread {thread_id}:")
            MessageInfo.print_conversation(thread_root.root_message)

        # order thread_roots by depth and number of reactions and get to 10
        thread_roots = dict(
            sorted(
                thread_roots.items(),
                key=lambda item: (item[1].depth, item[1].total_reactions),
                reverse=True,
            )[:limit]
        )

        # add each thread info to highlights
        highlights = ""
        for thread_id, thread_root in thread_roots.items():
            highlights += f"\n {thread_root}\n"
            llm_summary = self.summarize_thread(
                thread_root, channel_name, context_info="thread"
            )
            highlights += f"   Summary:\n {llm_summary}\n"
        return highlights

    def apply_filters(self, message, filters):
        return True
        # Implement filtering logic based on keywords or other criteria
        if "keywords" in filters:
            message_text = message.text.lower() if message.text else ""
            for keyword in filters["keywords"]:
                if keyword.lower() in message_text:
                    return True
            return False  # Skip messages that don't contain the keywords
        return True  # No filters applied

    def get_user_str(self, message: Message) -> str:
        if message.from_user:
            name = message.from_user.first_name or ""
            if name and message.from_user.last_name:
                name += " " + message.from_user.last_name
            username = message.from_user.username or ""
            return f"{name} (@{username})".strip()
        else:
            return "Unknown User"

    # def get_user_url(self, message):

    def extract_links(self, text):
        if not text:
            return []
        return self.url_pattern.findall(text)

    async def extract_attachments(self, message, channel_name, date):
        attachments = []
        return attachments
        # Define the directory path for the channel and date
        channel_dir = os.path.join(self.attachments_dir, channel_name)
        date_dir = os.path.join(channel_dir, date.strftime("%Y-%m-%d"))
        os.makedirs(date_dir, exist_ok=True)

        # Check for different types of attachments
        if message.photo:
            file_path = await self.download_attachment(message, date_dir, "photo")
            if file_path:
                attachments.append({"type": "photo", "path": file_path})
        if message.document:
            file_path = await self.download_attachment(
                message, date_dir, message.document.file_name
            )
            if file_path:
                attachments.append({"type": "document", "path": file_path})
        if message.video:
            file_path = await self.download_attachment(message, date_dir, "video")
            if file_path:
                attachments.append({"type": "video", "path": file_path})
        # Add more attachment types as needed

        return attachments

    async def download_attachment(self, msg, directory, file_name):
        # Download the attachment and save it to the specified directory
        try:
            name = msg.link.split("/")[-1]
            file_name = os.path.join(directory, name)
            file_path = await self.app.download_media(
                msg,
                file_name=file_name,
                # file_ref=None,
                # progress=None,
                # progress_args=None,
            )
            if file_path:
                # Move the file to the desired directory
                final_path = os.path.join(directory, os.path.basename(file_path))
                os.rename(file_path, final_path)
                print(f"Downloaded attachment to {final_path}")
                return final_path
        except Exception as e:
            print(f"Error downloading attachment: {e}")
        return None

    def chunk_text(self, lines: list[str]) -> list[list[str]]:
        chunks = []
        current_chunk = []
        current_length = 0

        for idx, line in enumerate(lines):
            tokens = tokenizer.encode(line, add_special_tokens=False)
            token_length = len(tokens)

            if token_length > self.max_length:
                # Skip lines that are too long
                print(f"Skipping line {idx + 1} as it exceeds the max_length.")
                continue

            if current_length + token_length > self.max_length:
                # Start a new chunk
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [line]
                current_length = token_length
            else:
                current_chunk.append(line)
                current_length += token_length

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def summarize_chunk(self, chunk, channel_name):
        prompt = f"""
Summarize the key points and main topics discussed in the messages from the Telegram group titled "{channel_name}". Focus on the important details, actions required, decisions made, and any relevant information that stands out. Exclude small talk or irrelevant conversation, and aim to create a clear and concise summary that captures the essence of the discussions. Write summaries in a clear and concise manner. {format_style}
Here are the messages from the group:
{chunk}
"""
        try:
            return self.model.chat(prompt)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            return None

    def generate_summaries(
        self, msgs: list[str], channel_name: str, offset: datetime
    ) -> list[str]:
        chunks = self.chunk_text(msgs)

        summaries = []
        for idx, chunk in enumerate(chunks):
            chunk_input = "\n".join(chunk)
            print(f"Summarizing chunk {idx + 1} of {len(chunks)} for {channel_name}...")
            summary = self.summarize_chunk(chunk_input, channel_name)
            if summary:
                summaries.append(summary)
            else:
                summaries.append(f"Summary of chunk {idx + 1} could not be generated.")

        self.save_summaries(summaries, channel_name, offset)

        return self.summarize_summaries(summaries, offset)

    def get_dir_path(self, channel_name, offset):
        date_str = offset.strftime("%Y-%m-%d")
        channel_date_dir = os.path.join(self.output_dir, channel_name, date_str)
        os.makedirs(channel_date_dir, exist_ok=True)
        return channel_date_dir

    def save_summaries(self, summaries, channel_name, offset):
        output_file = os.path.join(
            self.get_dir_path(channel_name, offset), f"summary.txt"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for summary in summaries:
                f.write(summary + "\n")
        print(f"Summaries saved to {output_file}")
        # print summaries to console
        for summary in summaries:
            print(summary)

    def save_urls(self, urls, channel_name, offset):
        if not urls:
            return
        output_file = os.path.join(self.get_dir_path(channel_name, offset), f"urls.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for url in urls:
                f.write(url + "\n")
        print(f"URLs saved to {output_file}")

    def save_attachments_info(self, attachments, channel_name, offset):
        if not attachments:
            return
        output_file = os.path.join(
            self.get_dir_path(channel_name, offset), f"attachments.txt"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for attachment in attachments:
                f.write(f"Type: {attachment['type']}, Path: {attachment['path']}\n")
        print(f"Attachments info saved to {output_file}")

    @staticmethod
    def split_message(message, max_length=4096):
        return [message[i : i + max_length] for i in range(0, len(message), max_length)]

    # Modified send_summary_to_channel function
    async def send_summary_to_channel(
        self, summary: str, entities: list[MessageEntity] = None
    ):
        channel_id = "me"

        # summary = markdown.markdown(summary)

        # Split the summary into parts
        parts = self.split_message(summary)

        # Send each part separately
        for part in parts:
            await self.app.send_message(
                chat_id=channel_id,
                text=part,
                # entities=entities,
                # parse_mode=ParseMode.DISABLED,
            )

    def summarize_summaries(self, summaries: list[str], offset: datetime) -> str:
        # if summaries are too long, summarize them
        if len(summaries) <= 1:
            return summaries[0]

        # insert offset time into first summary

        # split the string into tokens and again ask model to summarize

        prompt = f"You are an assistant that summarizes conversations. Summarize the following summaries. Write summaries in a clear and concise manner. {format_style}\n{'\n'.join(summaries)}"

        try:
            return self.model.chat(prompt)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            return ""


if __name__ == "__main__":
    summarizer = TelegramSummarizer()
    # summarizer.app.run(summarizer.fetch_dialogs())
    summarizer.app.run(summarizer.process_channels())
