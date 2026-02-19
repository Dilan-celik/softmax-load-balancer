Softmax Load Balancer
Dağıtık sistemler için Softmax Action Selection algoritması kullanan yük dengeleyici.
Problem
Bir kümede K adet sunucu var. Sunucu performansları zamanla değişiyor (non-stationary). Round-Robin ve Random gibi klasik algoritmalar sunucu performansını öğrenemez.
Bu proje, Reinforcement Learning prensiplerine dayanan Softmax algoritmasını kullanarak sunucu performansından öğrenen adaptif bir çözüm sunar.
Özellikler

Softmax (Boltzmann) Action Selection
Exponential Moving Average (EMA) ile non-stationary adaptasyon
Log-Sum-Exp trick ile nümerik stabilite
Temperature decay (exploration → exploitation)
3 algoritma karşılaştırması (Softmax, Round-Robin, Random)
12 JUnit test
ASCII görselleştirme

Sonuçlar
Softmax algoritması 2000 istek üzerinde:

Round-Robin'e göre %25.6 daha hızlı
Random'a göre %31.4 daha hızlı
Ortalama gecikme: 42.1 ms (Round-Robin: 56.6 ms, Random: 61.4 ms)

Matematiksel Temel
Softmax Formülü
P(sunucu_i) = exp(Q_i / τ) / Σ exp(Q_j / τ)

Q_i: Sunucu i'nin tahmini ödülü
τ (tau): Sıcaklık parametresi

Nümerik Stabilite
Log-Sum-Exp trick ile overflow engellenir:
M = max(Q_i / τ)
P_i = exp(Q_i/τ - M) / Σ exp(Q_j/τ - M)
EMA Güncelleme
Q_i ← (1 - α) × Q_i + α × reward_i
reward_i = -latency_i / 100
Kurulum
Gereksinimler

Java 17+
Maven 3.8+
IntelliJ IDEA

Çalıştırma
bashgit clone https://github.com/Dilan-celik/softmax-load-balancer.git
cd softmax-load-balancer
mvn compile
mvn test
mvn exec:java -Dexec.mainClass="com.loadbalancer.Main"
IntelliJ'de

File → Open → Proje klasörünü seç
Main.java → Yeşil ▶ buton

Proje Yapısı
src/
├── main/java/com/loadbalancer/
│   ├── Main.java                    # Ana program
│   ├── algorithm/
│   │   ├── SoftmaxLoadBalancer.java # Ana algoritma
│   │   ├── RoundRobinLoadBalancer.java
│   │   └── RandomLoadBalancer.java
│   ├── model/
│   │   ├── Server.java
│   │   └── Request.java
│   ├── metrics/
│   │   └── MetricsCollector.java
│   ├── simulation/
│   │   └── Simulation.java
│   └── ui/
│       └── ConsoleVisualizer.java
└── test/java/com/loadbalancer/
    └── LoadBalancerTest.java        # 12 test
Test Sonuçları
Tests run: 12, Failures: 0, Errors: 0, Skipped: 0
Testler:

Olasılık dağılımı doğrulaması
Nümerik stabilite
Q-değeri yakınsaması
Temperature etkileri
Simülasyon bütünlüğü

Çalışma Zamanı

Round-Robin: O(1)
Random: O(1)
Softmax: O(K) - K sunucu sayısı
Bellek: O(K)

Neden Softmax?

ε-greedy: Rastgele keşif, Q-değerlerini göz ardı eder
Softmax: Q-değerleri orantılı keşif yapar
Sample average: Tüm geçmişe eşit ağırlık
EMA: Eski verilere daha az ağırlık (non-stationary için ideal)
