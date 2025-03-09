# YouTube 자막 요약 도구

YouTube 동영상의 자막을 추출하고 GPT-4o-mini 모델을 사용하여 요약해주는 파이썬 기반 도구입니다.

## 기능

- YouTube 동영상 URL을 입력받아 자막을 추출
- 자막이 공개되어 있을 경우 youtube_transcript_api를 사용
- 자막이 없는 경우 YouTube Data API v3를 활용하여 캡션 정보 조회 및 다운로드
- 추출된 자막을 GPT-4o-mini 모델을 사용하여 요약
- 결과를 텍스트 파일로 저장

## 설치 방법

1. 저장소를 클론합니다:
   ```
   git clone [저장소 URL]
   cd youtube_transcript
   ```

2. 필요한 패키지를 설치합니다:
   ```
   pip install -r requirements.txt
   ```

3. API 키를 설정합니다:
   - `.env.example` 파일을 `.env`로 복사합니다:
     ```
     # Windows
     copy .env.example .env
     
     # Linux/Mac
     cp .env.example .env
     ```
   - `.env` 파일을 열고 API 키를 입력합니다:
     ```
     # YouTube API 키 (선택적)
     YOUTUBE_API_KEY=your_youtube_api_key_here
     
     # OpenAI API 키 (필수)
     OPENAI_API_KEY=your_openai_api_key_here
     ```
   - YouTube Data API v3 키: [Google Cloud Console](https://console.cloud.google.com/)에서 발급
   - OpenAI API 키: [OpenAI 웹사이트](https://platform.openai.com/)에서 발급

## 사용 방법

1. 실행 스크립트를 사용하여 프로그램을 실행합니다:
   ```
   # Windows
   run.bat
   
   # Linux/Mac
   ./run.sh
   ```
   또는 직접 Python으로 실행:
   ```
   python main.py
   ```

2. 프롬프트에 YouTube 동영상 URL을 입력합니다.

3. 처리가 완료되면 `result_YYYYMMDD_HHMMSS.txt` 형식의 파일에 결과가 저장됩니다.

## 주의사항

- YouTube Data API v3는 일일 할당량이 제한되어 있습니다.
- 매우 긴 동영상의 경우 자막 텍스트가 너무 길어 요약이 어려울 수 있습니다.
- 일부 동영상은 자막이 없거나 접근이 제한되어 있을 수 있습니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 