import os
import subprocess
from dotenv import load_dotenv

def huggingface_login():
    """
    .env 파일에서 Hugging Face 토큰을 읽어 CLI 로그인을 실행합니다.
    """
    # 현재 디렉토리의 .env 파일에서 환경 변수를 불러옵니다.
    load_dotenv()

    # .env 파일에 정의된 HUGGING_FACE_TOKEN 값을 가져옵니다.
    hf_token = os.getenv("HUGGING_FACE_TOKEN")

    # 토큰 값이 있는지 확인합니다.
    if not hf_token:
        print("오류: .env 파일에서 'HUGGING_FACE_TOKEN'을 찾을 수 없습니다.")
        print("   .env 파일이 스크립트와 같은 위치에 있는지, 변수 이름이 올바른지 확인하세요.")
        return

    try:
        print("Hugging Face CLI 로그인을 시도합니다...")
        
        # huggingface-cli login 명령어 실행
        # ['huggingface-cli', 'login', '--token', 토큰값] 형태로 명령어를 구성합니다.
        command = ['huggingface-cli', 'login', '--token', hf_token]
        
        # subprocess.run을 사용하여 외부 명령어 실행
        # check=True: 명령어가 실패하면 예외 발생
        # capture_output=True: 실행 결과를 캡처
        # text=True: 결과를 텍스트(문자열)로 다룸
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True
        )
        
        print("로그인 성공!")
        # 로그인 성공 시 출력되는 메시지를 확인합니다.
        # print(result.stdout) 
        
    except FileNotFoundError:
        print("오류: 'huggingface-cli' 명령어를 찾을 수 없습니다.")
        print("   'pip install huggingface_hub'로 라이브러리가 올바르게 설치되었는지 확인하세요.")
    except subprocess.CalledProcessError as e:
        print("로그인 실패:")
        print(f"   오류 메시지: {e.stderr.strip()}")
    except Exception as e:
        print(f"   예상치 못한 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    huggingface_login()