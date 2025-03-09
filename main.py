#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YouTube 자막 요약 도구
- YouTube 동영상 URL을 입력받아 자막을 추출하고 요약하는 도구
- 자막이 공개되어 있을 경우 youtube_transcript_api를 사용
- 자막이 없는 경우 YouTube Data API v3를 활용하여 캡션 정보 조회 및 다운로드
- 추출된 자막을 GPT-4o-mini 모델을 사용하여 요약
- 결과를 텍스트 파일로 저장
"""

import re
import os
import sys
import json
import logging
import datetime
import requests
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from functools import wraps
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, NoTranscriptAvailable
from dotenv import load_dotenv
import tqdm  # 진행률 표시를 위한 tqdm 추가

# .env 파일 로드
load_dotenv()

# result 폴더 생성 (없는 경우)
result_dir = "result"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    print(f"'{result_dir}' 폴더를 생성했습니다.")

# 로그 파일 경로 설정
log_file = os.path.join(result_dir, "youtube_transcript.log")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),  # 인코딩을 utf-8로 설정
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("youtube_transcript")

# 불필요한 로그 제외
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# API 키 설정 (.env 파일에서 가져옴)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# 상수 정의
DEFAULT_LANGUAGES = ["ko", "en"]  # 기본 자막 언어 우선순위
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MODEL_NAME = "gpt-4o-mini"  # 요약에 사용할 모델

# API 사용량 추적을 위한 변수
api_usage = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_cost_usd": 0.0
}

# OpenAI API 가격 정보 (2024년 8월 기준)
# 출처: https://platform.openai.com/settings/organization/limits
OPENAI_PRICING = {
    "gpt-4o-mini": {
        "input_per_1k": 0.15,  # 달러/1K 토큰
        "output_per_1k": 0.60,  # 달러/1K 토큰
        "exchange_rate": 1450.0  # 달러 대 원화 환율 (2024년 8월 기준)
    }
}

# 예외 클래스 정의
class YouTubeTranscriptError(Exception):
    """YouTube 자막 요약 도구에서 발생하는 기본 예외 클래스"""
    pass

class VideoIDError(YouTubeTranscriptError):
    """동영상 ID 추출 관련 오류"""
    pass

class TranscriptError(YouTubeTranscriptError):
    """자막 추출 관련 오류"""
    pass

class APIError(YouTubeTranscriptError):
    """API 호출 관련 오류"""
    pass

class SummaryError(YouTubeTranscriptError):
    """요약 관련 오류"""
    pass

# 유틸리티 함수
def handle_error(func: Callable) -> Callable:
    """
    함수에서 발생하는 예외를 처리하는 데코레이터
    
    Args:
        func: 데코레이트할 함수
        
    Returns:
        데코레이트된 함수
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except YouTubeTranscriptError as e:
            logger.error(f"{func.__name__} 실행 중 오류 발생: {e}")
            print(f"오류: {e}")
            return None
        except Exception as e:
            logger.error(f"{func.__name__} 실행 중 예상치 못한 오류 발생: {e}", exc_info=True)
            print(f"예상치 못한 오류가 발생했습니다: {e}")
            return None
    return wrapper

def check_api_keys() -> Dict[str, bool]:
    """
    필요한 API 키가 설정되어 있는지 확인하는 함수
    
    Returns:
        API 키 설정 상태를 담은 딕셔너리
    """
    # API 키가 설정되어 있는지 확인
    youtube_api_valid = bool(YOUTUBE_API_KEY) and YOUTUBE_API_KEY != "your_youtube_api_key_here"
    openai_api_valid = bool(OPENAI_API_KEY) and OPENAI_API_KEY != "your_openai_api_key_here"
    
    return {
        "youtube_api": youtube_api_valid,
        "openai_api": openai_api_valid
    }

def validate_url(url: str) -> bool:
    """
    URL이 유효한 YouTube URL인지 확인하는 함수
    
    Args:
        url: 확인할 URL
        
    Returns:
        유효한 YouTube URL이면 True, 아니면 False
    """
    youtube_patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?(?=.*v=([0-9A-Za-z_-]{11}))',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([0-9A-Za-z_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([0-9A-Za-z_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    
    return False

def exit_with_error(message: str, exit_code: int = 1) -> None:
    """
    오류 메시지를 출력하고 프로그램을 종료하는 함수
    
    Args:
        message: 출력할 오류 메시지
        exit_code: 종료 코드 (기본값: 1)
    """
    logger.error(message)
    print(f"오류: {message}")
    sys.exit(exit_code)

def parse_srt_content(srt_content: str) -> List[Dict]:
    """
    SRT 형식의 자막 내용을 파싱하는 함수
    
    Args:
        srt_content: SRT 형식의 자막 내용
        
    Returns:
        파싱된 자막 데이터 리스트 (각 항목은 시작 시간, 종료 시간, 텍스트를 포함하는 딕셔너리)
    """
    # 자막 블록 분리를 위한 정규식
    pattern = r'(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3})\s+([\s\S]*?)(?=\n\d+\s+\d{2}:\d{2}:\d{2},\d{3}|$)'
    
    matches = re.findall(pattern, srt_content)
    
    subtitles = []
    for match in matches:
        index, start_time, end_time, text = match
        
        # 텍스트 정리 (여러 줄 -> 한 줄, 앞뒤 공백 제거)
        text = re.sub(r'\s+', ' ', text).strip()
        
        subtitles.append({
            'index': int(index),
            'start': start_time,
            'end': end_time,
            'text': text
        })
    
    return subtitles

def srt_to_text(srt_content: str) -> str:
    """
    SRT 형식의 자막 내용을 단일 텍스트로 변환하는 함수
    
    Args:
        srt_content: SRT 형식의 자막 내용
        
    Returns:
        모든 자막 텍스트를 결합한 문자열
    """
    subtitles = parse_srt_content(srt_content)
    return ' '.join([subtitle['text'] for subtitle in subtitles])

# 주요 기능 함수
@handle_error
def extract_video_id(url: str) -> Optional[str]:
    """
    YouTube URL에서 동영상 ID를 추출하는 함수
    
    Args:
        url: YouTube 동영상 URL
        
    Returns:
        추출된 동영상 ID 또는 None (추출 실패 시)
    """
    # YouTube URL 패턴에서 동영상 ID 추출을 위한 정규식
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # 일반 YouTube URL
        r'(?:embed\/)([0-9A-Za-z_-]{11})',  # 임베드 URL
        r'(?:shorts\/)([0-9A-Za-z_-]{11})',  # 쇼츠 URL
        r'youtu\.be\/([0-9A-Za-z_-]{11})'   # 단축 URL
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # 유효한 동영상 ID를 찾지 못한 경우
    logger.warning(f"유효한 YouTube 동영상 ID를 찾을 수 없습니다: {url}")
    raise VideoIDError(f"유효한 YouTube 동영상 ID를 찾을 수 없습니다: {url}")

@handle_error
def get_video_info(video_id: str) -> Dict:
    """
    YouTube Data API를 사용하여 동영상 정보를 가져오는 함수
    
    Args:
        video_id: YouTube 동영상 ID
        
    Returns:
        동영상 정보를 담은 딕셔너리
    """
    # YouTube API 키가 없으면 기본 정보 반환
    if not YOUTUBE_API_KEY:
        logger.warning("YouTube API 키가 설정되지 않았습니다. 동영상 정보를 가져올 수 없습니다.")
        return {"title": "알 수 없음", "description": "알 수 없음"}
    
    try:
        # API 요청 URL 구성
        url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={YOUTUBE_API_KEY}&part=snippet"
        
        # API 요청 전송
        response = requests.get(url)
        response.raise_for_status()
        
        # 응답 데이터 처리
        data = response.json()
        if not data.get("items"):
            logger.warning(f"동영상 정보를 찾을 수 없습니다: {video_id}")
            return {"title": "알 수 없음", "description": "알 수 없음"}
        
        # 동영상 정보 추출
        snippet = data["items"][0]["snippet"]
        
        return {
            "title": snippet.get("title", "알 수 없음"),
            "description": snippet.get("description", "알 수 없음"),
            "publishedAt": snippet.get("publishedAt", "알 수 없음"),
            "channelTitle": snippet.get("channelTitle", "알 수 없음")
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"YouTube API 요청 중 오류 발생: {e}")
        # 오류 발생 시 기본 정보 반환
        return {"title": "알 수 없음", "description": "알 수 없음"}
    except Exception as e:
        logger.error(f"동영상 정보를 가져오는 중 오류 발생: {e}")
        raise APIError(f"동영상 정보를 가져오는 중 오류 발생: {e}")

@handle_error
def get_transcript_with_api(video_id: str, languages: List[str] = None) -> Optional[str]:
    """
    youtube_transcript_api를 사용하여 자막을 가져오는 함수
    
    Args:
        video_id: YouTube 동영상 ID
        languages: 자막 언어 우선순위 리스트
        
    Returns:
        결합된 자막 텍스트 또는 None (자막 없음)
    """
    if languages is None:
        languages = DEFAULT_LANGUAGES
    
    try:
        # 자막 목록 가져오기
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # 우선순위에 따라 자막 언어 선택
        transcript = None
        for lang in languages:
            try:
                transcript = transcript_list.find_transcript([lang])
                logger.info(f"'{lang}' 언어로 자막을 찾았습니다.")
                break
            except NoTranscriptFound:
                continue
        
        # 우선순위 언어가 없으면 자동 생성된 자막 시도
        if transcript is None:
            try:
                # 자동 생성된 자막 중 첫 번째 것 선택
                transcript = next(transcript_list)
                logger.info(f"자동 생성된 자막을 사용합니다. 언어: {transcript.language}")
            except StopIteration:
                logger.warning(f"동영상 {video_id}에 사용 가능한 자막이 없습니다.")
                return None
        
        # 자막 가져오기
        transcript_data = transcript.fetch()
        
        # 자막 텍스트 결합
        transcript_text = ' '.join([item['text'] for item in transcript_data])
        
        return transcript_text
        
    except (TranscriptsDisabled, NoTranscriptAvailable) as e:
        logger.warning(f"youtube_transcript_api로 자막을 가져올 수 없습니다: {e}")
        return None
    except Exception as e:
        logger.error(f"자막을 가져오는 중 오류 발생: {e}")
        raise TranscriptError(f"자막을 가져오는 중 오류 발생: {e}")

@handle_error
def get_transcript_with_youtube_api(video_id: str) -> Optional[str]:
    """
    YouTube Data API v3를 사용하여 자막을 가져오는 함수
    
    Args:
        video_id: YouTube 동영상 ID
        
    Returns:
        결합된 자막 텍스트 또는 None (자막 없음)
    """
    # YouTube API 키가 없으면 None 반환
    if not YOUTUBE_API_KEY:
        logger.warning("YouTube API 키가 설정되지 않았습니다.")
        return None
    
    try:
        # 1. 캡션 목록 조회
        captions_url = f"https://www.googleapis.com/youtube/v3/captions?videoId={video_id}&key={YOUTUBE_API_KEY}&part=snippet"
        response = requests.get(captions_url)
        response.raise_for_status()
        
        captions_data = response.json()
        if not captions_data.get("items"):
            logger.warning(f"해당 동영상({video_id})에 캡션이 없습니다.")
            return None
        
        # 2. 캡션 ID 찾기 (한국어 우선, 없으면 영어)
        caption_id = None
        caption_language = None
        
        for caption in captions_data["items"]:
            lang = caption["snippet"]["language"]
            if lang == "ko":
                caption_id = caption["id"]
                caption_language = "ko"
                break
            elif lang == "en" and caption_id is None:
                caption_id = caption["id"]
                caption_language = "en"
        
        if caption_id is None:
            # 언어 상관없이 첫 번째 캡션 선택
            caption_id = captions_data["items"][0]["id"]
            caption_language = captions_data["items"][0]["snippet"]["language"]
        
        logger.info(f"캡션 ID를 찾았습니다. 언어: {caption_language}")
        
        # 3. 캡션 다운로드 (OAuth 2.0 인증 필요)
        # 참고: 캡션 다운로드는 OAuth 2.0 인증이 필요하므로 API 키만으로는 불가능할 수 있음
        # 이 경우 대안으로 youtube-dl 등의 도구를 사용하거나 다른 방법을 고려해야 함
        
        # 임시 방편으로 간단한 예시 구현 (실제로는 작동하지 않을 수 있음)
        download_url = f"https://www.googleapis.com/youtube/v3/captions/{caption_id}?key={YOUTUBE_API_KEY}"
        headers = {
            "Authorization": f"Bearer {YOUTUBE_API_KEY}"  # 실제로는 OAuth 2.0 토큰이 필요
        }
        
        try:
            download_response = requests.get(download_url, headers=headers)
            download_response.raise_for_status()
            
            # 임시 파일에 SRT 내용 저장
            with tempfile.NamedTemporaryFile(suffix='.srt', delete=False, mode='w', encoding='utf-8') as temp_file:
                temp_file.write(download_response.text)
                temp_file_path = temp_file.name
            
            # SRT 파일 파싱하여 텍스트 추출
            srt_content = download_response.text
            transcript_text = srt_to_text(srt_content)
            
            # 임시 파일 삭제
            os.unlink(temp_file_path)
            
            return transcript_text
            
        except Exception as e:
            logger.error(f"캡션 다운로드 중 오류 발생: {e}")
            logger.warning("YouTube Data API v3로 캡션을 다운로드하려면 OAuth 2.0 인증이 필요할 수 있습니다.")
            return None
    
    except requests.exceptions.RequestException as e:
        logger.error(f"YouTube API 요청 중 오류 발생: {e}")
        return None
    except Exception as e:
        logger.error(f"YouTube API로 자막을 가져오는 중 오류 발생: {e}")
        return None

@handle_error
def summarize_text(text: str) -> Optional[str]:
    """
    OpenAI API를 사용하여 텍스트를 요약하는 함수
    
    Args:
        text: 요약할 텍스트
        
    Returns:
        요약된 텍스트 또는 None (요약 실패 시)
    """
    # OpenAI API 키가 없으면 None 반환
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API 키가 설정되지 않았습니다.")
        raise APIError("OpenAI API 키가 설정되지 않았습니다.")
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        # 요약 프롬프트 작성 (개선된 버전)
        prompt = f"""다음 YouTube 동영상 자막을 요약해주세요. 

요약 시 다음 사항을 반드시 포함해주세요:
1. 전체 내용의 핵심 주제와 목적
2. 주요 토픽별 세부 내용 (시간 순서대로)
3. 언급된 기술적 요소, 방법론, 도구 등이 있다면 구체적으로 명시
4. 중요한 결론이나 인사이트

요약은 3,000 토큰 이내로 작성해주세요. 너무 짧지 않게 충분한 정보를 담되, 중요하지 않은 세부 사항은 생략해주세요.

자막 내용:
{text}"""
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "당신은 YouTube 동영상 자막을 요약하는 전문가입니다. 전체 내용을 포괄적으로 이해하고, 주요 토픽과 기술적 요소를 구체적으로 포함하여 명확하게 요약해주세요."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 3000
        }
        
        logger.info(f"OpenAI API 요청 중... (모델: {MODEL_NAME})")
        
        # API 요청
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        summary = result["choices"][0]["message"]["content"]
        
        # 토큰 사용량 추적
        if "usage" in result:
            api_usage["input_tokens"] += result["usage"]["prompt_tokens"]
            api_usage["output_tokens"] += result["usage"]["completion_tokens"]
            
            # 로그에 사용량 기록
            logger.info(f"API 사용량: 입력 토큰 {result['usage']['prompt_tokens']}개, 출력 토큰 {result['usage']['completion_tokens']}개")
        
        return summary
    
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenAI API 요청 중 오류 발생: {e}")
        raise APIError(f"OpenAI API 요청 중 오류 발생: {e}")
    except Exception as e:
        logger.error(f"텍스트 요약 중 오류 발생: {e}")
        raise SummaryError(f"텍스트 요약 중 오류 발생: {e}")

@handle_error
def save_result(video_url: str, video_id: str, video_info: Dict, transcript: str, summary: str) -> str:
    """
    결과를 텍스트 파일로 저장하는 함수
    
    Args:
        video_url: YouTube 동영상 URL
        video_id: YouTube 동영상 ID
        video_info: 동영상 정보 딕셔너리
        transcript: 자막 텍스트
        summary: 요약 텍스트
        
    Returns:
        저장된 파일 경로
    """
    # result 폴더 확인 (전역 변수 사용)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        logger.info(f"'{result_dir}' 폴더를 생성했습니다.")
    
    # 현재 시간을 파일명에 포함
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(result_dir, f"result_{timestamp}.txt")
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("YouTube 동영상 자막 요약 결과\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("## 동영상 정보\n")
            f.write(f"URL: {video_url}\n")
            f.write(f"동영상 ID: {video_id}\n")
            f.write(f"제목: {video_info.get('title', '알 수 없음')}\n")
            f.write(f"채널: {video_info.get('channelTitle', '알 수 없음')}\n")
            f.write(f"게시일: {video_info.get('publishedAt', '알 수 없음')}\n\n")
            
            f.write("## 자막 정보\n")
            f.write(f"자막 길이: {len(transcript)} 글자\n\n")
            
            f.write("## 요약 결과\n")
            f.write(f"{summary}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"생성 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"결과가 '{filename}' 파일에 저장되었습니다.")
        return filename
    
    except Exception as e:
        logger.error(f"결과 저장 중 오류 발생: {e}")
        raise Exception(f"결과 저장 중 오류 발생: {e}")

def main():
    """
    메인 함수
    """
    print("=" * 80)
    print("YouTube 자막 요약 도구")
    print("=" * 80)
    
    # API 키 확인
    if not OPENAI_API_KEY:
        print("경고: OpenAI API 키가 설정되지 않았습니다. 요약 기능을 사용할 수 없습니다.")
        exit_with_error("OpenAI API 키가 필요합니다. .env 파일에 OPENAI_API_KEY를 설정해주세요.")
    
    # 사용자로부터 YouTube URL 입력 받기
    video_url = input("\nYouTube 동영상 URL을 입력하세요: ")
    
    # URL 유효성 검사
    if not validate_url(video_url):
        exit_with_error("유효한 YouTube URL이 아닙니다.")
    
    # 동영상 ID 추출
    video_id = extract_video_id(video_url)
    if not video_id:
        exit_with_error("유효한 YouTube 동영상 ID를 추출할 수 없습니다.")
    
    print(f"동영상 ID: {video_id}")
    
    # 전체 작업 진행률을 표시할 tqdm 객체 생성
    print("\n작업 진행 상황:")
    with tqdm.tqdm(total=100, desc="전체 진행률", 
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}] {postfix}",
                  postfix="준비 중...") as progress_bar:
        
        # 1. 동영상 정보 가져오기 (20%)
        progress_bar.set_postfix_str("동영상 정보 가져오는 중...")
        video_info = get_video_info(video_id)
        
        # video_info가 None인 경우 기본값 설정
        if video_info is None:
            video_info = {"title": "알 수 없음", "description": "알 수 없음"}
        
        print(f"\n동영상 제목: {video_info.get('title', '알 수 없음')}")
        progress_bar.update(20)  # 20% 진행
        
        # 2. 자막 가져오기 (30%)
        progress_bar.set_postfix_str("자막 가져오는 중...")
        transcript = get_transcript_with_api(video_id)
        
        # youtube_transcript_api로 자막을 가져올 수 없는 경우 YouTube Data API 사용
        if transcript is None and YOUTUBE_API_KEY:
            print("\nyoutube_transcript_api로 자막을 가져올 수 없어 YouTube Data API를 사용합니다...")
            transcript = get_transcript_with_youtube_api(video_id)
        
        if transcript is None:
            exit_with_error("자막을 가져올 수 없습니다. 해당 동영상에 자막이 없거나 접근이 제한되어 있을 수 있습니다.")
        
        print(f"\n자막 길이: {len(transcript)} 글자")
        progress_bar.update(30)  # 추가 30% 진행 (총 50%)
        
        # 3. 자막 요약하기 (40%)
        progress_bar.set_postfix_str("자막 요약하는 중...")
        
        # 요약 시작 시간 기록
        summary_start_time = time.time()
        
        # 요약 진행 상황을 실시간으로 업데이트하기 위한 함수
        def update_progress():
            elapsed = time.time() - summary_start_time
            while elapsed < 30:  # 최대 30초 동안 업데이트
                # 경과 시간에 따라 진행률 업데이트 (최대 35%까지)
                progress = min(35, elapsed * 1.2)  # 초당 약 1.2%씩 증가
                progress_bar.set_postfix_str(f"자막 요약하는 중... ({elapsed:.1f}초 경과)")
                progress_bar.n = 50 + progress  # 50%에서 시작
                progress_bar.refresh()
                time.sleep(0.5)
                elapsed = time.time() - summary_start_time
                
                # 요약이 완료되었으면 중단
                if hasattr(update_progress, 'completed') and update_progress.completed:
                    break
        
        # 별도 스레드에서 진행률 업데이트 실행
        import threading
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        # 실제 요약 수행
        summary = summarize_text(transcript)
        
        # 요약 완료 표시
        update_progress.completed = True
        progress_thread.join(timeout=1)
        
        # 진행률 강제 업데이트
        progress_bar.n = 90  # 90%로 설정
        progress_bar.set_postfix_str("요약 완료")
        progress_bar.refresh()
        
        if summary is None:
            exit_with_error("자막을 요약할 수 없습니다. API 키를 확인하거나 나중에 다시 시도해주세요.")
        
        # 4. 결과 저장하기 (10%)
        progress_bar.set_postfix_str("결과 저장하는 중...")
        filename = save_result(video_url, video_id, video_info, transcript, summary)
        
        # 완료
        progress_bar.update(10)  # 마지막 10% 진행 (총 100%)
        progress_bar.set_postfix_str("작업 완료!")
    
    print("\n처리가 완료되었습니다.")
    print(f"결과 파일: {filename}")
    
    # 결과 파일 경로를 절대 경로로 변환하여 표시
    abs_path = os.path.abspath(filename)
    print(f"절대 경로: {abs_path}")
    
    # API 사용량 및 비용 표시
    if api_usage["input_tokens"] > 0 or api_usage["output_tokens"] > 0:
        # 입력 및 출력 토큰 비용 계산
        input_cost_usd = (api_usage["input_tokens"] / 1000) * OPENAI_PRICING[MODEL_NAME]["input_per_1k"]
        output_cost_usd = (api_usage["output_tokens"] / 1000) * OPENAI_PRICING[MODEL_NAME]["output_per_1k"]
        total_cost_usd = input_cost_usd + output_cost_usd
        
        # 원화 환산
        exchange_rate = OPENAI_PRICING[MODEL_NAME]["exchange_rate"]
        input_cost_krw = input_cost_usd * exchange_rate
        output_cost_krw = output_cost_usd * exchange_rate
        total_cost_krw = total_cost_usd * exchange_rate
        
        print("\n" + "=" * 60)
        print("OpenAI API 사용량 및 비용")
        print("=" * 60)
        print(f"모델: {MODEL_NAME}")
        print(f"입력 토큰: {api_usage['input_tokens']:,}개 (${input_cost_usd:.4f}, ₩{input_cost_krw:.0f})")
        print(f"출력 토큰: {api_usage['output_tokens']:,}개 (${output_cost_usd:.4f}, ₩{output_cost_krw:.0f})")
        print(f"총 토큰: {api_usage['input_tokens'] + api_usage['output_tokens']:,}개")
        print("-" * 40)
        print(f"총 비용 (USD): ${total_cost_usd:.4f}")
        print(f"총 비용 (KRW): ₩{total_cost_krw:.0f}")
        print("-" * 40)
        print(f"가격 정보 출처: https://platform.openai.com/settings/organization/limits")
        print(f"환율: $1 = ₩{exchange_rate:.0f} (2024년 8월 기준)")
        print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {e}", exc_info=True)
        print(f"\n오류가 발생했습니다: {e}")
        sys.exit(1) 