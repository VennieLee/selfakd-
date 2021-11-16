#self distillation #fakd 쓸 때,
171줄 train은 weight 파일이 있고 학생과 선생이 있는 코드를 돌릴 때 사용(ex. RCAN실험)
242줄 trainWeight는 weight 파일이 없는 SRResnet16 때문에 weight 파일 만들 때 사용했음
310줄 trainSRResnet은 self distilation실험 전용으로 선생 불러오는 코드랑 feature 받아들이는게 달라서 따로 선언해서 사용중

이 세개 중 골라서 main에서 train 함수를 바꿔줘야하고,
490줄 main은 선생, 학생이 있는 코드의 경우 train과 test하는 코드이고
519줄 main은 self distillation이나 weight 파일 만들 때 사용하는(선생이 없는) 코드이다.

option에 모든 argument가 들어가도록 지정했으니 그 부분을 수정해서 사용하면 됨.

loss 같은 경우 original loss와 forweightloss가 있는데 loss 파일에 사용할 loss를 복붙해서 썼음 => 깔끔하게 정리할 예정
