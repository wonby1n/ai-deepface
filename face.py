from deepface import DeepFace
import json
import os
import glob

def verify_identity(passport_image_path, selfie_image_path):
    """
    여권 사진과 현재 사진을 비교하여 본인 여부를 확인하는 함수입니다.
    
    Args:
        passport_image_path (str): 여권 사진 파일 경로
        selfie_image_path (str): 현재 사진(셀카) 파일 경로
        
    Returns:
        dict: 검증 결과 (성공 여부, 유사도 거리 등)
    """
    print("사진 비교를 시작합니다. 모델을 로드하는 데 시간이 걸릴 수 있습니다...")
    
    try:
        # DeepFace.verify를 사용하여 두 이미지 비교
        # model_name: 'ArcFace', 'Facenet512', 'VGG-Face' 등이 있으며, ArcFace가 동양인 및 까다로운 조건에서 성능이 우수한 편입니다.
        # enforce_detection: True로 설정 시 사진에서 얼굴을 찾지 못하면 예외(Exception)를 발생시킵니다.
        result = DeepFace.verify(
            img1_path=passport_image_path,
            img2_path=selfie_image_path,
            model_name="ArcFace", 
            detector_backend="retinaface", # 얼굴 탐지 성능이 뛰어난 retinaface 사용
            enforce_detection=True 
        )
        
        # 결과 분석
        is_verified = result.get('verified', False)
        distance = result.get('distance', 0.0)
        threshold = result.get('threshold', 0.0)
        
        print("\n=== [검증 결과] ===")
        if is_verified:
            print(f"✅ 동일인으로 확인되었습니다. (거리: {distance:.4f} / 기준치: {threshold})")
        else:
            print(f"❌ 다른 사람으로 판단됩니다. (거리: {distance:.4f} / 기준치: {threshold})")
            
        return result

    except ValueError as ve:
        # 얼굴을 인식하지 못한 경우의 예외 처리
        print(f"\n⚠️ 오류 발생: 사진에서 얼굴을 찾을 수 없습니다. ({str(ve)})")
        return {"error": "Face not detected", "details": str(ve)}
    
    except Exception as e:
        # 기타 예외 처리
        print(f"\n⚠️ 알 수 없는 오류 발생: {str(e)}")
        return {"error": "Unknown error", "details": str(e)}

# === 실행 예시 ===
if __name__ == "__main__":
    # 현재 스크립트(face.py)가 위치한 디렉토리의 절대 경로 계산
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 지정된 구조에 맞춘 폴더 경로 설정
    passport_dir = os.path.join(base_dir, "passport_img")
    selfie_dir = os.path.join(base_dir, "selfie_img")
    
    # 실용적인 접근: 폴더가 없으면 에러를 내지 않고 안내 메시지 출력
    if not os.path.exists(passport_dir) or not os.path.exists(selfie_dir):
        print(f"⚠️ 폴더가 존재하지 않습니다. 스크립트와 같은 위치에 아래 폴더를 생성해주세요.")
        print(f" - {passport_dir}")
        print(f" - {selfie_dir}")
    else:
        # 폴더 내의 첫 번째 이미지를 자동으로 가져오도록 처리
        passport_files = glob.glob(os.path.join(passport_dir, "*.*"))
        selfie_files = glob.glob(os.path.join(selfie_dir, "*.*"))
        
        if not passport_files:
            print(f"❌ '{passport_dir}' 폴더에 여권 이미지가 없습니다.")
        elif not selfie_files:
            print(f"❌ '{selfie_dir}' 폴더에 셀카 이미지가 없습니다.")
        else:
            # 첫 번째로 탐색된 파일을 지정
            passport_path = passport_files[0]
            selfie_path = selfie_files[0]
            
            print(f"📄 사용된 여권 이미지: {os.path.basename(passport_path)}")
            print(f"📄 사용된 셀카 이미지: {os.path.basename(selfie_path)}")
            
            # 함수 호출
            verification_result = verify_identity(passport_path, selfie_path)
            
            print("\n[상세 데이터]")
            print(json.dumps(verification_result, indent=4, ensure_ascii=False))