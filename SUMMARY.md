# YouTube 자막 요약 도구 - 프로젝트 요약

## 프로젝트 개요

이 프로젝트는 YouTube 동영상의 자막을 추출하고 GPT-4o-mini 모델을 사용하여 요약해주는 파이썬 기반 도구입니다. 사용자가 YouTube 동영상 URL을 입력하면 자막을 가져와 요약하고, 결과를 텍스트 파일로 저장합니다.

## 주요 기능

1. **YouTube URL 처리**
   - 다양한 형식의 YouTube URL에서 동영상 ID 추출
   - URL 유효성 검사

2. **자막 추출**
   - youtube_transcript_api를 사용한 자막 추출 (기본 방법)
   - YouTube Data API v3를 사용한 자막 추출 (대체 방법)
   - 다국어 자막 지원 (한국어 우선, 영어 대체)

3. **텍스트 요약**
   - OpenAI API를 사용하여 GPT-4o-mini 모델로 자막 요약
   - 요약 결과 포맷팅

4. **결과 저장**
   - 타임스탬프가 포함된 파일명으로 결과 저장
   - 동영상 정보, 자막 정보, 요약 결과를 포함한 결과 파일 생성

## 프로젝트 구조

```
youtube_transcript/
├── main.py              # 메인 스크립트 (모든 기능 통합)
├── .env.example         # API 키 설정 예제 파일
├── .env                 # API 키 설정 파일 (git에서 제외됨)
├── requirements.txt     # 의존성 패키지 목록
├── README.md            # 프로젝트 설명서
├── SUMMARY.md           # 프로젝트 요약
├── run.bat              # Windows 실행 스크립트
├── run.sh               # Linux/Mac 실행 스크립트
└── .gitignore           # Git 제외 파일 목록
```

## 기술 스택

- **언어:** Python 3.x
- **주요 라이브러리:**
  - youtube-transcript-api: YouTube 자막 추출
  - requests: HTTP 요청 처리
  - python-dotenv: 환경 변수 관리
- **외부 API:**
  - YouTube Data API v3: 동영상 정보 및 자막 조회
  - OpenAI API: 텍스트 요약 (GPT-4o-mini 모델)

## 오류 처리

- 사용자 정의 예외 클래스를 통한 체계적인 오류 처리
- 로깅 시스템을 통한 오류 기록
- 데코레이터 패턴을 활용한 일관된 오류 처리
- 사용자 친화적인 오류 메시지 제공

## 사용 방법

1. API 키 설정:
   - `.env.example` 파일을 `.env`로 복사
   - `.env` 파일에 API 키 입력 (YouTube API 키는 선택적, OpenAI API 키는 필수)

2. 의존성 패키지 설치:
   ```
   pip install -r requirements.txt
   ```

3. 프로그램 실행:
   - Windows: `run.bat` 실행
   - Linux/Mac: `./run.sh` 실행

4. YouTube 동영상 URL 입력

5. 결과 확인:
   - 생성된 `result_YYYYMMDD_HHMMSS.txt` 파일 확인

## 향후 개선 사항

1. **웹 인터페이스 개발**
   - Flask 또는 FastAPI를 사용한 웹 애플리케이션 개발
   - 사용자 친화적인 UI 제공

2. **자막 처리 개선**
   - 더 많은 언어 지원
   - 자막 번역 기능 추가

3. **요약 기능 강화**
   - 다양한 요약 모델 지원
   - 요약 길이 및 스타일 옵션 제공

4. **배치 처리 기능**
   - 여러 동영상 URL을 한 번에 처리하는 기능
   - 결과를 CSV 또는 JSON 형식으로 내보내기

## 결론

이 프로젝트는 YouTube 동영상의 자막을 쉽게 요약할 수 있는 도구를 제공합니다. 사용자는 동영상 URL만 입력하면 자동으로 자막을 추출하고 요약하여 결과를 저장할 수 있습니다. 오류 처리와 로깅 시스템을 통해 안정적인 동작을 보장하며, 향후 웹 인터페이스 및 추가 기능을 통해 더욱 발전할 수 있는 기반을 마련했습니다. 