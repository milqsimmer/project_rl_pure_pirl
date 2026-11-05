python train_rl.py --mode pure --seed 0 --steps 2000000 --learning-rate 3e-4 --lambda-a 0.01 --success-bonus 5.0 --max-delta-theta 0.2 --run-tag md02


--------------------------------
1) Config base que vamos usar para o pirl
--------------------------------
Mantemos tudo igual ao pure baseline, mudando só mode e alpha_tau:
- mode = pirl
- steps = 2_000_000
- learning-rate = 3e-4
- lambda_a = 0.01
- success_bonus = 5.0
- max-delta-theta = 0.2
- max-steps eval = 200
- target_seed default = seed (já está assim no código agora)
- Reward do pirl: igual ao pure, mas com o termo por passo
  - - alpha_tau * tau_sum ativo a cada step.
--------------------------------
2) Piloto: duas seeds, três valores de α_tau
--------------------------------
Resumo dos três valores de alpha_tau que você testou para pirl (seed 0), comparados com o pure baseline:
Pure baseline (md02, seed 0):
- success_rate = 0.98
- mean_final_dist ≈ 0.0612
- mean_tau_effort ≈ 2.02
- mean_energy ≈ 1.53
Pirl, alpha_tau = 0.0002 (md02_at2e4):
- success_rate = 0.835
- mean_final_dist ≈ 0.111
- mean_tau_effort ≈ 1.51  (↓ vs 2.02)
- mean_energy ≈ 1.45     (levemente ↓)
Pirl, alpha_tau = 0.0003 (md02_at3e4):
- success_rate = 0.745
- mean_final_dist ≈ 0.133
- mean_tau_effort ≈ 2.12  (↑ vs pure)
- mean_energy ≈ 2.13      (↑ vs pure)
Pirl, alpha_tau = 0.0005 (md02_at5e4):
- success_rate = 0.655
- mean_final_dist ≈ 0.181
- mean_tau_effort ≈ 1.51  (≈ igual ao 0.0002, ↓ vs pure)
- mean_energy ≈ 1.67      (↑ vs pure)
Leitura:
- alpha_tau = 0.0002:
  - Ainda mantém uma taxa de sucesso relativamente alta (0.835 vs 0.98 do pure).
  - Reduz torque médio de 2.02 → 1.51 (~25% a menos).
  - Energia também um pouco menor.
  - Esse é o melhor compromisso entre sucesso e redução de esforço entre os três.
- alpha_tau = 0.0003:
  - Cai sucesso (0.745) e PI-reward está forçando mais esforço: torque e energia maiores que o pure.  
    Não vale a pena.
- alpha_tau = 0.0005:
  - Sucesso pior (0.655), torque parecido com 0.0002, energia maior que o pure.  
    Pior trade-off que 0.0002.
Então, para a seed 0 com o setup atual (termo de torque por passo), alpha_tau = 0.0002 é claramente o candidato:
- success_rate_pirl ≈ 0.835 (aceitável, ainda alto),
- mean_tau_effort_pirl bem menor que o do pure.