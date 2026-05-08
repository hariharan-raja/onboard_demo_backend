import json
import os
import re
from datetime import datetime
from openai import AzureOpenAI, OpenAI

# Initialize OpenAI client
api_key_s=os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=api_key_s)
client = AzureOpenAI(
    api_key=api_key_s,
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

def match_location_to_id(transcript_text):
    for loc in location_data:
        desc = loc.get("locationDesc", "").lower()
        if desc in transcript_text.lower():
            return loc.get("locationId")
    return None

def extract_name_and_cabin(transcript):
    prompt = (
        f"From the following transcript: '{transcript}', extract the cabin number, if available. "
        "Return the result in JSON format like: {\"cabin\": \"<cabin number>\"}. "
        "If any of the information is not mentioned, return null for that field."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You extract name and cabin number from transcripts."},
                {"role": "user", "content": prompt}
            ]
        )

        response_text = response.choices[0].message.content.strip()
        response_text = re.sub(r"```json|```", "", response_text).strip()

        result = json.loads(response_text)
        return {"cabin": result.get("cabin")}
    except Exception as e:
        print(f"[extract_name_and_cabin] Error decoding: {e}")
        return {"cabin": None}

def load_guest_data():
    if os.path.exists("sample_guests.json"):
        with open("sample_guests.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {"passengerInfo": []}    

guest_data = load_guest_data()

def get_guest_details(cabin_number):
    for guest in guest_data.get("passengerInfo", []):
        if guest.get("cabin") == cabin_number:
            return {
                "firstName": guest.get("firstName"),
                "lastName": guest.get("lastName")
            }
    return None

def detect_guest_emotion(transcript):
    prompt = (
        f"Analyze the following conversation transcript: '{transcript}'. "
        "Identify the primary emotion expressed by the guest. "
        "Choose only from the following categories: angry, sad, neutral, satisfied, very-happy. " 
        "Return the result strictly in this JSON format: {\"emotion\": \"<detected emotion>\"}."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You analyze guest emotions from the transcript."},
                {"role": "user", "content": prompt}
            ]
        )

        response_text = response.choices[0].message.content.strip()
        response_text = re.sub(r"```json|```", "", response_text).strip()

        result = json.loads(response_text)
        return result.get("emotion", "Unknown")
    except Exception as e:
        print(f"[detect_guest_emotion] Error decoding emotion: {e}")
        return "Unknown"

def load_issue_data():
    if os.path.exists(ISSUE_DATA_FILE):
        with open(ISSUE_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

issue_data = load_issue_data()

def check_issue_match(transcript):
    issues_dict = {
        issue["issueTypeDesc"].strip().lower(): {
            "priorityDesc": issue["priorityDesc"],
            "issueGroupDesc": issue.get("issueGroupDesc"),
            "level1DepartmentDesc": issue.get("level1DepartmentDesc"),
            "issueTypeId": issue.get("issueTypeId"),
        }
        for issue in issue_data
        if "issueTypeDesc" in issue and "priorityDesc" in issue and "level2DepartmentDesc" in issue
    }

    issues_list = list(issues_dict.keys())

    prompt_text = (
        f"""You will receive a transcript snippet from a conversation along with a predefined list of known issues: {issues_list}.

    Your task is to carefully analyze the provided transcript snippet and identify if it **explicitly and unambiguously matches exactly ONE issue** from the given issues list. 

    Important rules:
    - If the transcript snippet is incomplete, unclear, ambiguous, or can match more than one issue from the list, you **MUST** set the matching issue as `null`.
    - Do NOT guess or infer an issue if the conversation hasn't explicitly mentioned the complete issue yet. Partial matches must return `null`.
    - Do NOT return any physical location descriptions; just return the issue type if explicitly matched.
    - Briefly summarize the provided transcript snippet in 2-3 sentences.
    - If compensation is mentioned, verify carefully if it has actually been confirmed or explicitly denied/retracted. 
        - Return the confirmed compensation ONLY if clearly confirmed.
        - If compensation is denied, unclear, or retracted, set compensation as `null`.

    Always return the result strictly in valid JSON format, like this:

    {{
        "issueTypeDesc": "<matching issue or null>",
        "priorityDesc": "<priority>",
        "level1DepartmentDesc": "<level1DepartmentDesc>",
        "summary": "<concise 2-3 sentence summary of the transcript>",
        "compensation": "<confirmed compensation or null>"
    }}

    Now, analyze the following transcript snippet carefully:
    '{transcript}'
    """
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You match user complaints to known issues."},
                {"role": "user", "content": prompt_text}
            ]
        )

        match_text = response.choices[0].message.content.strip()
        match_text = re.sub(r"```json|```", "", match_text).strip()
        match_result = json.loads(match_text)
        print("matchresult------>",match_result)

        summary_text = match_result.get("summary", "")
        issue_type_raw = match_result.get("issueTypeDesc")
        issue_type = issue_type_raw.strip().lower() if issue_type_raw else None
        matched_location_id = match_location_to_id(transcript)

        if issue_type not in issues_dict:
            return {
                "IssueType": None,
                "PriorityDesc": None,
                "IssueGroupDesc": None,
                "level1DepartmentDesc": None,
                "issueTypeId": None,
                "LocationId": matched_location_id,
                "Summary": None,
                "Compensation": None
            }

        return {
            "IssueType": issue_type_raw,
            "PriorityDesc": issues_dict[issue_type]["priorityDesc"],
            "IssueGroupDesc": issues_dict[issue_type]["issueGroupDesc"],
            "level1DepartmentDesc": issues_dict[issue_type]["level1DepartmentDesc"],
            "issueTypeId": issues_dict[issue_type]["issueTypeId"],
            "LocationId": matched_location_id,
            "Summary": summary_text,
            "Compensation": match_result.get("compensation", None)
        }

    except Exception as e:
        print(f"[check_issue_match] Error: {e}")
        return {
            "IssueType": None,
            "PriorityDesc": None,
            "IssueGroupDesc": None,
            "level1DepartmentDesc": None,
            "issueTypeId": None,
            "LocationId": None,
            "Summary": None,
            "Compensation": None
        }

def speaker_diarization(transcript):
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

def process_transcript(transcript):
    original_transcript = transcript
    labeled_transcript = speaker_diarization(transcript)

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": labeled_transcript
    }

    match_result = check_issue_match(original_transcript)
    guest_emotion = detect_guest_emotion(original_transcript) if match_result.get("IssueType") else None
    name_cabin_info = extract_name_and_cabin(original_transcript)

    guest_details = {}
    if name_cabin_info.get("cabin"):
        guest_details = get_guest_details(name_cabin_info["cabin"]) or {}

    combined_result = {
        "issueTypeId": match_result.get("issueTypeId"),
        "issueTypeDesc": match_result.get("IssueType"),
        "priorityDesc": match_result.get("PriorityDesc"),
        "IssueGroupDesc": match_result.get("IssueGroupDesc"),
        "level1DepartmentDesc": match_result.get("level1DepartmentDesc"),
        "cabin": name_cabin_info.get("cabin"),
        "guestDetails": guest_details,
        "locationId": match_result.get("LocationId"),
        "guestEmotion": guest_emotion,
        "summary": match_result.get("Summary"),
        "compensation": match_result.get("Compensation")
    }
    return labeled_transcript, combined_result
