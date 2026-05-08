import json
import os
import re
from datetime import datetime
from openai import AzureOpenAI
from rapidfuzz import fuzz

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint="https://oceanai-openai.openai.azure.com/"
)

ISSUE_DATA_FILE = "issue_data.json"

def load_location_data():
    if os.path.exists("Location.json"):
        with open("Location.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

location_data = load_location_data()


async def match_location_to_desc(transcript_text):
    for loc in location_data:
        # Skip entries that are guest or crew cabins
        if loc.get("guestCabin") or loc.get("crewCabin"):
            continue

        desc = loc.get("locationDesc", "").lower()
        if desc and desc in transcript_text.lower():
            # Return locationDesc 
            return loc.get("locationDesc")
    
    # No location matched
    return None


def load_guest_data():
    if os.path.exists("sample_guests.json"):
        with open("sample_guests.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {"passengerInfo": []}    

guest_data = load_guest_data()


async def get_guest_details(cabin_number, first_name=None, last_name=None):
    matching_guests = [
        guest for guest in guest_data.get("passengerInfo", [])
        if guest.get("cabin") == cabin_number
    ]

    if not matching_guests:
        return None

    best_match = None
    best_score = 0

    # If first or last name is provided, try to match using similarity
    if first_name or last_name:
        for guest in matching_guests:
            guest_fn = guest.get("firstName", "").lower()
            guest_ln = guest.get("lastName", "").lower()

            fn_score = fuzz.ratio(first_name.lower(), guest_fn) if first_name else 100
            ln_score = fuzz.ratio(last_name.lower(), guest_ln) if last_name else 100

            avg_score = (fn_score + ln_score) / 2

            if avg_score > best_score:
                best_score = avg_score
                best_match = guest

        if best_score >= 80:  # You can adjust this threshold
            return {
                "firstName": best_match.get("firstName"),
                "lastName": best_match.get("lastName")
            }

    # Fallback to first guest if no strong match found
    first_guest = matching_guests[0]
    return {
        "firstName": first_guest.get("firstName"),
        "lastName": first_guest.get("lastName")
    }


def load_issue_data():
    if os.path.exists(ISSUE_DATA_FILE):
        with open(ISSUE_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

issue_data = load_issue_data()

async def analyze_transcript_full(transcript, issues_list):
    prompt = f"""
You are a transcript analysis assistant. You will receive a conversation transcript between a Guest Services Officer and a guest.

Your job is to analyze the transcript and return the following details in a single valid JSON response:

1. **Name and Cabin**:
    - Extract guest's cabin number. Example --> even if cabin **11542** is said as eleven thousand five hundred forty two, return 11542. 
    - Extract guest's first name and last name if mentioned.
    - If not mentioned, return null for those fields.

2. **Guest Emotion**:
    - Detect the primary emotion expressed by the guest.
    - Choose only from: angry, sad, neutral, satisfied, very-happy.

3. **Issue Matching**:
    - You are given a list of known issues: {issues_list}.
    - If the transcript **explicitly and unambiguously** matches **only ONE** issue, return it.
    - If unclear, ambiguous, or multiple matches possible, set issueTypeDesc to null.
    - Return priorityDesc and level1DepartmentDesc only if issue is matched.

4. **Compensation**:
    - Review the conversation to determine if guest services have offered any compensation to address issues experienced by the guest. If a compensation (e.g., a bottle of champagne) is offered and accepted by the guest, document the compensation provided.

    - If the guest accepts the offered compensation, return that information.

    - If no compensation is offered or accepted, return null.

5. **Summary**:
    - Write a short summary (max 10 lines) of the conversation. Capture the information shared by both the guest and Guest Services Officer

The JSON structure should look like this:

{{
    "cabin": "<cabin number or null>",
    "firstName": "<guest first name or null>",
    "lastName": "<guest last name or null>",
    "emotion": "<one of: angry, sad, neutral, satisfied, very-happy>",
    "issueTypeDesc": "<matched issue from list or null>",
    "priorityDesc": "<priority if matched or null>",
    "level1DepartmentDesc": "<department if matched or null>",
    "compensation": "<confirmed compensation or null>",
    "summary": "<summary>"
}}

Now analyze the following transcript:
\"\"\"
{transcript}
\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You extract structured data from guest conversation transcripts."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r"```json|```", "", content).strip()
        return json.loads(content)

    except Exception as e:
        print(f"[analyze_transcript_full] Error: {e}")
        return {
            "cabin": None,
            "firstName": None,
            "lastName": None,
            "emotion": None,
            "issueTypeDesc": None,
            "priorityDesc": None,
            "level1DepartmentDesc": None,
            "compensation": None,
            "summary": None
        }




async def speaker_diarization(transcript):
    prompt_diarization = (
        f"Given the transcript text: '{transcript}', add speaker diarization to the transcript. The conversation is taking place between a Guest State officer (name might be provided in the transcript) and a guest (name might be provided in the transcript)"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You identify speakers in transcript."},
                {"role": "user", "content": prompt_diarization}
            ]
        )

        diarization_text = response.choices[0].message.content.strip()
        # print("Diarization Text:", diarization_text)
        return diarization_text

    except Exception as e:
        print(f"[speaker_diarization] Error: {e}")
        return transcript

async def process_transcript(transcript):
    original_transcript = transcript
    # labeled_transcript = await(speaker_diarization(transcript))

    issues_dict = {
        issue["issueTypeDesc"].strip().lower(): {
            "priorityDesc": issue["priorityDesc"],
            "issueGroupDesc": issue.get("issueGroupDesc"),  
            "level1DepartmentDesc": issue.get("level1DepartmentDesc"),
            "issueTypeId": issue.get("issueTypeId"),
        }
        for issue in issue_data if "issueTypeDesc" in issue and "priorityDesc" in issue and "level1DepartmentDesc" in issue
    }
    # print("issues_dict----------->",issues_dict)
    issues_list = list(issues_dict.keys())
    analysis = await (analyze_transcript_full(original_transcript, issues_list))
    issue_type = analysis.get("issueTypeDesc", "").strip().lower() if analysis.get("issueTypeDesc") else None
    # print("issue_type----------->",issue_type)
    issue_info = issues_dict.get(issue_type, {}) if issue_type else {}
    # Use the updated function to get the location description
    matched_location_desc = await(match_location_to_desc(original_transcript))
    if not matched_location_desc and analysis.get("cabin"):
        matched_location_desc = analysis["cabin"]
    # Guest info logic
    guest_details = {}
    if analysis.get("cabin"):
        guest_details = await(get_guest_details(
            analysis.get("cabin"),
            analysis.get("firstName"),
            analysis.get("lastName")
        )) or {}

    combined_result = {
        "issueTypeId": issue_info.get("issueTypeId"),
        "issueTypeDesc": analysis.get("issueTypeDesc"),
        "priorityDesc": issue_info.get("priorityDesc"),
        "IssueGroupDesc": issue_info.get("issueGroupDesc"),
        "level1DepartmentDesc": issue_info.get("level1DepartmentDesc"),
        "cabin": analysis.get("cabin"),
        "guestDetails": guest_details,
        "locationId": matched_location_desc,
        "guestEmotion": analysis.get("emotion"),
        "summary": analysis.get("summary"),
        "compensation": analysis.get("compensation")
    }
    # print("combined_resulttt------>",combined_result)
    return combined_result

async def convert_non_null_values_to_text(data):
    if isinstance(data, dict):
        return {k: await convert_non_null_values_to_text(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [await convert_non_null_values_to_text(item) for item in data]
    else:
        return str(data) if data is not None else None
def merge_transcriptions_with_timestamps(previous_segments, current_segments):
    """
    Merge two segment-based transcripts, trimming the overlap using timestamps.
    Avoids repeating content from previous segments in the current batch.
    """
    if not previous_segments:
        return " ".join(seg.get("text", "").strip() for seg in current_segments)

    # Get the last timestamp from previous segment
    last_prev_end = previous_segments[-1].get("end", 0)

    # Filter current segments: only include those that start *after* last end
    filtered_current = [seg for seg in current_segments if seg.get("start", 0) >= last_prev_end]

    merged_text = []
    merged_text.extend(seg.get("text", "").strip() for seg in previous_segments)
    merged_text.extend(seg.get("text", "").strip() for seg in filtered_current)

    return " ".join(merged_text).strip()