from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
import PyPDF2
import json
import io
import requests
import re
import tempfile
from Validations import (
    extract_text_excluding_tables,
    validate_text,
    contains_actual_sentences,
    validate_structure
)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Enable CORS for React frontend
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4.1')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION', '2024-08-01-preview')


def call_azure_openai(messages, temperature=0.7, max_tokens=2000):
    """Call Azure OpenAI API for completions"""
    try:
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
            raise Exception("Azure OpenAI credentials not configured")

        # Remove trailing slash from endpoint if present
        endpoint = AZURE_OPENAI_ENDPOINT.rstrip('/')

        # Construct the Azure OpenAI endpoint URL
        url = f"{endpoint}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"

        print(f"[DEBUG] Calling Azure OpenAI:")
        print(f"[DEBUG] URL: {url}")
        print(f"[DEBUG] Deployment: {AZURE_OPENAI_DEPLOYMENT}")
        print(f"[DEBUG] API Version: {AZURE_API_VERSION}")

        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY
        }

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120)

        print(f"[DEBUG] Response Status: {response.status_code}")

        if response.status_code != 200:
            error_data = response.json() if response.text else {}
            print(f"[DEBUG] Error Response: {error_data}")

            if response.status_code == 404:
                raise Exception(
                    f"Azure OpenAI deployment not found. "
                    f"Please check:\n"
                    f"1. Endpoint: {endpoint}\n"
                    f"2. Deployment name: {AZURE_OPENAI_DEPLOYMENT}\n"
                    f"3. API version: {AZURE_API_VERSION}\n"
                    f"Error: {error_data.get('error', {}).get('message', response.text)}"
                )
            else:
                raise Exception(f"Azure OpenAI API error ({response.status_code}): {error_data.get('error', {}).get('message', response.text)}")

        data = response.json()
        return data['choices'][0]['message']['content']

    except requests.exceptions.Timeout:
        raise Exception("AI request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")
    except Exception as e:
        raise Exception(f"AI API call failed: {str(e)}")


def extract_topic_from_text(text):
    """Extract the main topic/subject from the text content"""
    print(f"[DEBUG] Sending text to Azure OpenAI ({len(text)} chars)")
    # Take first 2000 characters for topic analysis
    sample_text = text[:2000] if len(text) > 2000 else text

    prompt = f"""Analyze this educational content and identify the main topic/subject in 1-3 words ONLY.

Content:
{sample_text}

Examples:
- "Photosynthesis"
- "World War II"
- "Algebraic Equations"
- "Cell Biology"

Respond with ONLY the topic name (1-3 words), nothing else."""

    messages = [
        {
            "role": "system",
            "content": "You are a topic identifier. Respond ONLY with the main topic name in 1-3 words."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        topic = call_azure_openai(messages, temperature=0.3, max_tokens=50)
        # Clean up the response
        topic = topic.strip().strip('"').strip("'")
        return topic
    except Exception as e:
        print(f"Topic extraction failed: {str(e)}")
        # Fallback: extract first meaningful line
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                return line[:50]
        return "Unknown Topic"


# Original routes
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Flask API is working!",
        "service": "ExamBits AI Service",
        "version": "3.1.0 (Azure OpenAI + PDF Validation)"
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "ExamBits AI",
        "ai_service": "Azure OpenAI",
        "endpoint": AZURE_OPENAI_ENDPOINT,
        "deployment": AZURE_OPENAI_DEPLOYMENT,
        "api_version": AZURE_API_VERSION,
        "api_configured": bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY)
    })


@app.route("/api/ai/test-connection", methods=["GET"])
def test_connection():
    """Test Azure OpenAI connection"""
    try:
        messages = [
            {
                "role": "user",
                "content": "Say 'Connection successful' if you can read this."
            }
        ]

        response = call_azure_openai(messages, temperature=0.3, max_tokens=50)

        return jsonify({
            "success": True,
            "message": "Azure OpenAI connection successful",
            "response": response,
            "config": {
                "endpoint": AZURE_OPENAI_ENDPOINT,
                "deployment": AZURE_OPENAI_DEPLOYMENT,
                "api_version": AZURE_API_VERSION
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "config": {
                "endpoint": AZURE_OPENAI_ENDPOINT,
                "deployment": AZURE_OPENAI_DEPLOYMENT,
                "api_version": AZURE_API_VERSION
            }
        }), 500


@app.route("/api/ai/extract-pdf", methods=["POST"])
def extract_pdf():
    """Extract text content from uploaded PDF file with validation"""
    temp_pdf_path = None

    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400

        file = request.files['file']

        if not file.filename:
            return jsonify({"success": False, "error": "No file selected"}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"success": False, "error": "File must be a PDF"}), 400

        # Save uploaded file to temporary location for validation
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            file.save(temp_file.name)
            temp_pdf_path = temp_file.name

        print(f"[PDF Validation] Validating PDF structure: {file.filename}")

        # Step 1: Check if PDF has complex structure (tables + multi-column)
        if validate_structure(temp_pdf_path):
            print("[PDF Validation] PDF has complex structure (tables detected)")
            return jsonify({
                "success": False,
                "error": "This PDF has a complex structure with tables or multiple columns that cannot be processed accurately. Please upload a simpler PDF with plain text content."
            }), 400

        # Step 2: Extract text using advanced method (handles multi-column, excludes tables)
        print("[PDF Validation] Extracting text with column detection...")
        text = extract_text_excluding_tables(temp_pdf_path)

        if not text or text == False:
            print("[PDF Validation] Complex structure detected during extraction")
            return jsonify({
                "success": False,
                "error": "This PDF has a complex structure that cannot be processed. Please try a different PDF."
            }), 400

        # Step 3: Validate text has actual sentences
        if not contains_actual_sentences(text):
            print("[PDF Validation] No proper sentences detected")
            return jsonify({
                "success": False,
                "error": "The extracted text does not contain proper sentences. The PDF might be image-based or have an unusual format."
            }), 400

        # Step 4: Validate minimum sentence count
        is_valid_length, sentence_count = validate_text(text)
        if not is_valid_length:
            print(f"[PDF Validation] Text too short: {sentence_count} sentences")
            return jsonify({
                "success": False,
                "error": f"The PDF content is too short ({sentence_count} sentences). Please upload a PDF with more substantial content (minimum 10 sentences)."
            }), 400

        print(f"[PDF Validation] ✓ Valid PDF with {sentence_count} sentences")

        # Count pages from the extracted text
        pages = text.count('--- Page')

        return jsonify({
            "success": True,
            "content": text.strip(),
            "pages": pages,
            "char_count": len(text),
            "sentence_count": sentence_count,
            "validation": {
                "has_tables": False,
                "is_multi_column": False,
                "is_valid": True
            }
        })

    except Exception as e:
        print(f"[PDF Validation] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "PDF extraction failed",
            "details": str(e)
        }), 500

    finally:
        # Clean up temporary file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file: {e}")


@app.route("/api/ai/analyze-topic", methods=["POST"])
def analyze_topic():
    """Extract main topic from content"""
    try:
        data = request.json

        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        content = data.get('content', '')

        if not content or len(content.strip()) < 50:
            return jsonify({
                "success": False,
                "error": "Content is too short"
            }), 400

        print(f"Analyzing topic from {len(content)} characters...")

        topic = extract_topic_from_text(content)

        print(f"Extracted topic: {topic}")

        return jsonify({
            "success": True,
            "topic": topic,
            "content_length": len(content)
        })

    except Exception as e:
        print(f"Error in analyze_topic: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Topic analysis failed",
            "details": str(e)
        }), 500


@app.route("/api/ai/generate-questions", methods=["POST"])
def generate_questions():
    """Generate exam questions from full PDF content"""
    try:
        data = request.json

        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        content = data.get('content', '')
        num_questions = int(data.get('num_questions', 10))
        difficulty = data.get('difficulty', 'medium')
        question_type = data.get('type', 'multiple-choice')
        # Validation
        if not content or len(content.strip()) < 100:
            return jsonify({
                "success": False,
                "error": "Content is too short to generate questions"
            }), 400

        if num_questions < 1 or num_questions > 50:
            return jsonify({
                "success": False,
                "error": "Number of questions must be between 1 and 50"
            }), 400

        # Truncate content if too long (keep first ~8000 chars to stay within token limits)
        max_content_length = 8000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "\n\n[Content truncated for processing...]"

        print(f"Generating {num_questions} {question_type} questions from {len(content)} characters...")

        # Build strict format requirements based on question type
        if question_type == 'multiple-choice':
            format_rules = """MULTIPLE CHOICE FORMAT (STRICT):
- Exactly 4 options labeled A, B, C, D
- Options array must have exactly 4 strings
- correct_answer must be one letter: "A", "B", "C", or "D"
- Example: {"question": "What is X?", "options": ["Choice A", "Choice B", "Choice C", "Choice D"], "correct_answer": "B"}"""

        elif question_type == 'true-false':
            format_rules = """TRUE/FALSE FORMAT (STRICT):
- Exactly 2 options: ["True", "False"]
- correct_answer must be exactly "True" or "False"
- Example: {"question": "Statement?", "options": ["True", "False"], "correct_answer": "True"}"""

        else:  # identification
            format_rules = """IDENTIFICATION FORMAT (STRICT):
- Empty options array: []
- correct_answer is a short text answer (1-5 words)
- Example: {"question": "What is ___?", "options": [], "correct_answer": "Photosynthesis"}"""

        # Create the prompt
        prompt = f"""You are an expert exam question generator. Generate {num_questions} {difficulty} difficulty {question_type} questions based on the following educational content.

CONTENT:
{content}

REQUIREMENTS:
1. Generate EXACTLY {num_questions} questions
2. Difficulty level: {difficulty}
3. Question type: {question_type}
4. Questions must be based ONLY on information in the content above
5. Questions should test understanding, not just memorization
6. All questions must be clear and unambiguous

{format_rules}

CRITICAL: Return ONLY a valid JSON array. No markdown, no explanations, no additional text.
Format: [
  {{"question": "...", "options": [...], "correct_answer": "..."}},
  ...
]

Generate the questions now:"""

        messages = [
            {
                "role": "system",
                "content": f"You are an expert exam question generator. You create {question_type} questions based on educational content. You ONLY return valid JSON arrays with no additional formatting or text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Call Azure OpenAI
        response_text = call_azure_openai(messages, temperature=0.7, max_tokens=2000)

        if not response_text:
            raise Exception("No response received from AI model")

        print(f"AI Response received: {len(response_text)} characters")

        # Clean response - remove markdown code blocks if present
        response_text = response_text.strip()

        # Remove markdown code blocks
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()

        # Parse JSON
        try:
            questions = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {str(e)}")
            print(f"Response text: {response_text[:500]}")
            return jsonify({
                "success": False,
                "error": "Failed to parse AI response as JSON",
                "details": str(e)
            }), 500

        # Validate response
        if not isinstance(questions, list) or len(questions) == 0:
            return jsonify({
                "success": False,
                "error": "Invalid response format from AI"
            }), 500

        # Validate and clean each question
        validated_questions = []
        for i, q in enumerate(questions):
            if not isinstance(q, dict) or 'question' not in q or 'correct_answer' not in q:
                print(f"Question {i+1} missing required fields, skipping")
                continue

            # Type-specific validation
            options = q.get('options', [])
            is_valid = False

            if question_type == 'identification':
                # Must have empty or no options
                if not options or len(options) == 0:
                    is_valid = True
                    q['options'] = []  # Ensure empty array

            elif question_type == 'true-false':
                # Must have exactly 2 options: True and False
                if len(options) == 2:
                    # Normalize to ensure exact format
                    q['options'] = ["True", "False"]
                    # Normalize answer
                    if q['correct_answer'].lower() in ['true', 't', 'yes']:
                        q['correct_answer'] = "True"
                    else:
                        q['correct_answer'] = "False"
                    is_valid = True

            elif question_type == 'multiple-choice':
                # Must have exactly 4 options
                if len(options) == 4:
                    # Ensure answer is a single letter A-D
                    answer = q['correct_answer'].strip().upper()
                    if answer in ['A', 'B', 'C', 'D']:
                        q['correct_answer'] = answer
                        is_valid = True

            if not is_valid:
                print(f"Question {i+1} failed validation for type {question_type}, skipping")
                continue

            # Add validated question
            validated_questions.append({
                'question': q['question'],
                'options': q['options'],
                'correct_answer': q['correct_answer'],
                'type': question_type
            })

        if len(validated_questions) == 0:
            return jsonify({
                "success": False,
                "error": f"No valid {question_type} questions were generated. Please try again."
            }), 500

        print(f"Successfully validated {len(validated_questions)} questions")

        return jsonify({
            "success": True,
            "questions": validated_questions,
            "count": len(validated_questions)
        })

    except Exception as e:
        print(f"Error in generate_questions: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Question generation failed",
            "details": str(e)
        }), 500


if __name__ == "__main__":
        port = int(os.environ.get("PORT", 5000))
        app.run(host='0.0.0.0', port=port, debug=True)

#
# if __name__ == "__main__":
#     print("\n" + "="*60)
#     print("🚀 ExamBits AI Service Starting (Azure OpenAI + Validation)...")
#     print("="*60)
#
#     if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
#         print("❌ ERROR: Azure OpenAI credentials not configured")
#         print("📝 Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")
#         print("="*60)
#         exit(1)
#
#     print(f"✅ AI Service: Azure OpenAI")
#     print(f"✅ Endpoint: {AZURE_OPENAI_ENDPOINT}")
#     print(f"✅ Deployment: {AZURE_OPENAI_DEPLOYMENT}")
#     print(f"✅ API Version: {AZURE_API_VERSION}")
#     print(f"✅ Server: http://localhost:5000")
#     print(f"✅ PDF Validation: Enabled (pdfplumber)")
#     print(f"✅ Test Connection: http://localhost:5000/api/ai/test-connection")
#     print("="*60 + "\n")
#
#     app.run(host='0.0.0.0', port=5000, debug=True)