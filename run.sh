#!/bin/bash

echo "YouTube 자막 요약 도구 실행"
echo "============================"
echo

# 필요한 패키지 설치 확인
echo "필요한 패키지를 확인하는 중..."
pip install -r requirements.txt

# .env 파일 확인
if [ ! -f .env ]; then
    echo "경고: .env 파일이 없습니다."
    echo ".env.example 파일을 .env로 복사하고 API 키를 설정해주세요."
    cp .env.example .env
    echo ".env 파일이 생성되었습니다. API 키를 설정한 후 다시 실행해주세요."
    echo
    read -p "계속하려면 아무 키나 누르세요..."
    exit 1
fi

echo
echo "프로그램을 실행합니다..."
echo

python main.py

echo
read -p "계속하려면 아무 키나 누르세요..." 