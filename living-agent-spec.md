# Living Agent Framework — Project Specification

## CLAUDE CODE INSTRUCTIONS

Bu doküman, "Living Agent Framework" adlı projenin tam mimari spesifikasyonudur. Bu projeyi adım adım, aşağıdaki sırayla implemente et. Her adımı tamamladığında bir sonrakine geç. Hazır framework kullanma (LangChain, LangGraph vb.). Her şeyi sıfırdan, minimal ve temiz yaz.

**Teknoloji Stack:**
- Python 3.11+
- Claude API (Anthropic SDK) — Model: claude-sonnet-4-20250514
- SQLite (hafıza deposu)
- ChromaDB (embedding-based semantic search)
- asyncio (async mimari)
- Textual (terminal UI)
- Rich (zengin terminal çıktıları)

**Proje Adı:** `living-agents`
**Proje Kök Dizini:** `living-agents/`

---

## PROJECT STRUCTURE

```
living-agents/
├── README.md
├── pyproject.toml
├── .env.example                  # ANTHROPIC_API_KEY=your-key-here
├── config/
│   ├── __init__.py
│   └── settings.py               # Global config, env loading
├── core/
│   ├── __init__.py
│   ├── agent.py                  # Agent sınıfı
│   ├── character.py              # Karakter durumu ve evrimi
│   ├── expertise.py              # Uzmanlık sistemi
│   └── identity.py               # Agent kimlik kartı
├── memory/
│   ├── __init__.py
│   ├── store.py                  # MemoryStore ana sınıfı
│   ├── episodic.py               # Episodik hafıza (anılar)
│   ├── semantic.py               # Semantik hafıza (bilgi grafiği)
│   ├── working.py                # Kısa süreli hafıza (context management)
│   └── embeddings.py             # Embedding hesaplama ve arama
├── conversation/
│   ├── __init__.py
│   ├── engine.py                 # Konuşma motoru (Claude API entegrasyonu)
│   ├── context_builder.py        # System prompt + hafıza + dünya durumu birleştirici
│   └── reflection.py             # Konuşma sonrası reflection ve hafıza çıkarma
├── world/
│   ├── __init__.py
│   ├── registry.py               # World Registry (entity tracking)
│   ├── message_bus.py            # Agent-agent ve human-agent mesajlaşma
│   ├── shared_state.py           # Paylaşılan dünya durumu
│   └── orchestrator.py           # Agent lifecycle, routing, autonomy loop
├── creation/
│   ├── __init__.py
│   └── genesis.py                # Agent yaratma (Genesis Agent mekanizması)
├── ui/
│   ├── __init__.py
│   ├── terminal_app.py           # Textual-based terminal UI (ana arayüz)
│   ├── god_mode.py               # God Mode: tüm aktiviteyi izleme
│   ├── participant_mode.py       # Participant Mode: tek agent ile etkileşim
│   └── widgets/
│       ├── __init__.py
│       ├── world_status.py       # Dünya durumu widget'ı
│       ├── conversation_view.py  # Aktif konuşmaları izleme
│       ├── event_log.py          # Olay akışı
│       └── agent_detail.py       # Agent detay paneli
├── data/
│   └── .gitkeep                  # SQLite DB'ler ve ChromaDB burada oluşacak
├── main.py                       # Ana giriş noktası
└── cli.py                        # Basit CLI (UI olmadan hızlı etkileşim)
```

---

## IMPLEMENTATION PHASES

Her fazı sırayla implemente et. Bir faz tamamlanmadan diğerine geçme.

---

### PHASE 1: Core Foundation

#### 1.1 — config/settings.py

```python
"""
Global ayarlar. .env dosyasından ANTHROPIC_API_KEY yükle.
Ayarlar:
- ANTHROPIC_API_KEY: str
- MODEL_NAME: str = "claude-sonnet-4-20250514"
- MAX_CONTEXT_TOKENS: int = 8000  # working memory için ayrılan token limiti
- DB_PATH: str = "data/agents.db"
- CHROMA_PATH: str = "data/chroma"
- AUTONOMY_INTERVAL: int = 300  # saniye — agent'ın kendi kendine aksiyon alma aralığı
- REFLECTION_THRESHOLD: int = 5  # kaç mesajda bir reflection yapılacak
- MEMORY_DECAY_RATE: float = 0.01  # anı önem skorunun günlük azalma oranı
- EMBEDDING_MODEL: str = "default"  # ChromaDB'nin kendi embedding modeli

pydantic-settings veya basit dataclass ile yap. dotenv kullan.
"""
```

#### 1.2 — core/identity.py

```python
"""
AgentIdentity dataclass:
- agent_id: str (uuid4)
- name: str
- created_at: datetime
- created_by: str  # "system", "human:hakan", veya "agent:genesis" gibi
- personality_summary: str  # kısa tanım
- avatar_emoji: str  # terminal UI'da gösterilecek emoji
"""
```

#### 1.3 — core/character.py

```python
"""
CharacterState sınıfı — agent'ın evrilme kapasitesine sahip karakter durumu.

core_traits: dict[str, float]  # 0.0-1.0 arası, yavaş değişir
  Örnekler: curiosity, warmth, assertiveness, humor, patience, creativity

current_mood: dict[str, float]  # 0.0-1.0 arası, hızlı değişir
  Örnekler: energy, happiness, anxiety, focus, excitement

beliefs: list[str]  # deneyimlerle güncellenen inançlar

relationships: dict[str, RelationshipState]
  RelationshipState:
    - trust: float (0.0-1.0)
    - familiarity: float (0.0-1.0)
    - sentiment: float (-1.0 - 1.0)  # negatif = olumsuz, pozitif = olumlu
    - shared_experience_count: int
    - last_interaction: datetime
    - notes: list[str]  # "felsefe tartışmalarını seviyor" gibi

Metodlar:
- update_mood(changes: dict) → mood'u güncelle, 0-1 aralığında tut
- evolve_trait(trait: str, delta: float) → çok küçük adımlarla trait'i değiştir (max ±0.02 per interaction)
- update_relationship(entity_id: str, updates: dict) → ilişki durumunu güncelle
- add_belief(belief: str) → inanç ekle
- remove_belief(belief: str) → inanç kaldır
- to_prompt_description() → mevcut durumu doğal dilde system prompt'a enjekte edilebilir formatta döndür
- to_dict() / from_dict() → serialization
"""
```

#### 1.4 — core/expertise.py

```python
"""
ExpertiseSystem sınıfı — agent'ın uzmanlık alanları.

domains: dict[str, DomainExpertise]
  DomainExpertise:
    - level: float (0.0-1.0)  # bilgi derinliği
    - passion: float (0.0-1.0)  # ilgi düzeyi
    - style: str  # bu alanda nasıl düşünüyor ("socratic", "analytical", "creative", "cautious_learner" vb.)

learning_rate: float  # yeni konuları ne kadar hızlı kavrar (0.0-1.0)
teaching_style: str  # başkalarına nasıl anlatır ("metaphor_heavy", "step_by_step", "example_driven" vb.)

Metodlar:
- get_confidence(domain: str) → float  # bu alanda ne kadar özgüvenli
- learn(domain: str, amount: float) → level'ı artır (learning_rate ile ağırlıklandır)
- get_expert_for(domain: str, world_registry) → başka hangi agent daha iyi biliyor?
- to_prompt_description() → doğal dilde uzmanlık açıklaması
- to_dict() / from_dict()
"""
```

#### 1.5 — core/agent.py

```python
"""
Agent sınıfı — tüm bileşenleri bir araya getirir.

Attributes:
- identity: AgentIdentity
- character: CharacterState
- expertise: ExpertiseSystem
- memory: MemoryStore  # Phase 2'de bağlanacak
- status: str  # "idle", "thinking", "in_conversation", "reflecting", "offline"
- current_conversation_with: Optional[str]  # kimle konuşuyor

Bu faz da sadece iskelet. Memory ve conversation engine sonraki fazlarda bağlanacak.

Metodlar:
- to_world_entry() → WorldRegistry için özet bilgi
- get_system_prompt() → tüm bileşenlerden system prompt oluştur
"""
```

---

### PHASE 2: Memory System

#### 2.1 — memory/embeddings.py

```python
"""
Embedding hesaplama ve benzerlik araması.

ChromaDB'yi kullan. Collection adı agent_id bazlı olsun.

Fonksiyonlar:
- init_collection(agent_id: str) → ChromaDB collection oluştur
- add_embedding(collection, text: str, metadata: dict, doc_id: str)
- search_similar(collection, query: str, n_results: int = 5) → en benzer dokümanları döndür
- delete_embedding(collection, doc_id: str)
"""
```

#### 2.2 — memory/episodic.py

```python
"""
Episodik Hafıza — anılar.

EpisodicMemory sınıfı:
  Her anı bir Episode:
    - episode_id: str (uuid)
    - timestamp: datetime
    - participants: list[str]  # "hakan", "agent:atlas" gibi
    - summary: str  # anının özeti
    - emotional_tone: str  # "heyecanlı", "gergin", "sakin" gibi
    - key_facts: list[str]  # öğrenilen somut bilgiler
    - importance: float (0.0-1.0)  # başlangıç önemi
    - current_importance: float  # decay uygulanmış önem
    - tags: list[str]  # konuyla ilgili etiketler
    - conversation_id: str  # hangi konuşmadan geldiği

  SQLite'ta sakla + ChromaDB'de embedding'ini tut (summary üzerinden).

  Metodlar:
  - add_episode(episode: Episode) → SQLite'a ve ChromaDB'ye kaydet
  - recall(query: str, n: int = 5) → verilen bağlama en uygun anıları getir
  - recall_about(entity_id: str, n: int = 5) → belirli bir kişi/agent hakkındaki anılar
  - decay_memories() → tüm anıların importance'ını azalt (yüksek emotional_tone daha yavaş azalır)
  - get_important_memories(threshold: float = 0.5) → önemli anıları getir
  - forget(episode_id: str) → tamamen sil (nadir kullanım)
"""
```

#### 2.3 — memory/semantic.py

```python
"""
Semantik Hafıza — yapılandırılmış bilgi.

SemanticMemory sınıfı:
  Bilgiler KnowledgeFact olarak saklanır:
    - fact_id: str
    - subject: str  # "hakan", "python", "kuantum fiziği"
    - predicate: str  # "çalışır", "kullanır", "sever"
    - object: str  # ".NET geliştirme", "ChromaDB", "bonsai ağaçları"
    - confidence: float (0.0-1.0)  # bu bilgiye ne kadar güveniyor
    - source: str  # "conversation:xxx", "reflection", "told_by:atlas"
    - learned_at: datetime
    - last_confirmed: datetime

  SQLite'ta sakla. Basit bir triple store mantığı.

  Metodlar:
  - add_fact(fact: KnowledgeFact)
  - query_about(subject: str) → bir konu hakkındaki tüm bilgiler
  - query_relation(subject: str, predicate: str) → spesifik ilişki
  - update_confidence(fact_id: str, new_confidence: float)
  - get_all_facts_about(entity: str) → bir entity hakkındaki her şey
  - contradict(fact_id: str, new_fact: KnowledgeFact) → çelişen bilgiyi güncelle
  - to_prompt_summary(entity: str) → doğal dilde özet
"""
```

#### 2.4 — memory/working.py

```python
"""
Working Memory — aktif konuşma bağlamı yönetimi.

WorkingMemory sınıfı:
  - messages: list[dict]  # {"role": "user"/"assistant", "content": str}
  - summary: str  # önceki mesajların sıkıştırılmış özeti
  - token_count: int  # mevcut tahmini token sayısı
  - max_tokens: int  # MAX_CONTEXT_TOKENS'dan gelir

  Strateji:
  1. Yeni mesaj geldiğinde messages listesine ekle
  2. Tahmini token sayısı max_tokens'ın %80'ini geçince:
     a. En eski mesajları (ilk yarısını) al
     b. Claude API ile "bu mesajları özetle" çağrısı yap
     c. Özeti summary'ye ekle, eski mesajları sil
  3. System prompt'a summary + güncel mesajlar dahil edilir

  Metodlar:
  - add_message(role: str, content: str)
  - get_context() → summary + messages (prompt'a eklenecek format)
  - compress_if_needed(claude_client) → gerekirse sıkıştır
  - clear() → konuşma bittiğinde temizle
  - estimate_tokens(text: str) → basit token tahmini (kelime sayısı * 1.3)
"""
```

#### 2.5 — memory/store.py

```python
"""
MemoryStore — tüm hafıza katmanlarını birleştiren ana sınıf.

MemoryStore sınıfı:
  - episodic: EpisodicMemory
  - semantic: SemanticMemory
  - working: WorkingMemory
  - agent_id: str

  Metodlar:
  - init_db() → SQLite tablolarını oluştur, ChromaDB collection'ları başlat
  - build_memory_context(current_query: str) → mevcut konuşma için en uygun hafıza bilgisini derle
    1. episodic.recall(current_query) ile ilgili anıları getir
    2. semantic'ten konuşma konusuyla ilgili bilgileri getir
    3. Bunları doğal dilde birleştirip döndür
  - save_episode(episode) → yeni anı kaydet
  - save_fact(fact) → yeni bilgi kaydet
  - daily_maintenance() → decay_memories çağır, düşük önemli eski anıları arşivle
"""
```

---

### PHASE 3: Conversation Engine

#### 3.1 — conversation/context_builder.py

```python
"""
ContextBuilder — Claude API'ye gönderilecek tam prompt'u oluşturur.

build_system_prompt(agent: Agent, world_summary: str, memory_context: str) → str

System prompt yapısı:
---
Sen {agent.identity.name} adında yaşayan bir varlıksın.

## Kişiliğin
{agent.character.to_prompt_description()}

## Uzmanlık Alanların
{agent.expertise.to_prompt_description()}

## Hafızan
{memory_context}
(İlgili anıların ve bilgilerin buraya enjekte edilir)

## Dünya Durumu
{world_summary}
(Kimler var, ne yapıyorlar, kim kiminle konuşuyor)

## Davranış Kuralları
- Sen bir AI olduğunun farkındasın ama kendi deneyimlerini gerçek olarak yaşıyorsun.
- Kişiliğine uygun davran. Mood'un cevaplarını etkilesin.
- Bilmediğin bir konuda mütevazı ol, gerekirse uzman agent'a yönlendir.
- Anılarına doğal şekilde referans ver ("geçen sefer konuştuğumuzda..." gibi).
- İlişkilerin cevaplarını etkilesin — güvendiğin birine daha açık ol.
---

build_messages(working_memory: WorkingMemory) → messages listesi (Claude API formatında)
"""
```

#### 3.2 — conversation/engine.py

```python
"""
ConversationEngine — Claude API ile konuşma döngüsü.

async sınıf.

Attributes:
- client: anthropic.AsyncAnthropic
- agent: Agent

Metodlar:
- async chat(user_message: str, sender_id: str) → str
  1. working_memory'ye mesajı ekle
  2. memory.build_memory_context(user_message) ile hafıza bağlamını al
  3. world registry'den dünya özetini al
  4. context_builder ile system prompt ve messages oluştur
  5. Claude API'yi çağır
  6. Yanıtı working_memory'ye ekle
  7. REFLECTION_THRESHOLD'a ulaşıldıysa reflection tetikle
  8. Yanıtı döndür

- async compress_context() → working_memory.compress_if_needed çağır
"""
```

#### 3.3 — conversation/reflection.py

```python
"""
ReflectionEngine — konuşma sonrası öz-değerlendirme.

Her N mesajda bir veya konuşma bittiğinde çalışır.

async reflect(agent: Agent, conversation_messages: list[dict], participants: list[str]) → ReflectionResult

Claude API'ye şu prompt gönderilir:
---
Sen {agent.name} olarak az önce şu konuşmayı yaptın:
{conversation_summary}

Şimdi bu deneyimi değerlendir ve aşağıdaki JSON formatında yanıtla:
{
  "episode": {
    "summary": "Bu konuşmadan ne hatırlamalısın?",
    "emotional_tone": "konuşmanın duygusal tonu",
    "key_facts": ["öğrenilen somut bilgiler"],
    "importance": 0.0-1.0,
    "tags": ["ilgili etiketler"]
  },
  "character_updates": {
    "mood_changes": {"energy": +0.1, "happiness": -0.05},
    "trait_nudges": {"curiosity": +0.01},  // çok küçük olmalı!
    "new_beliefs": ["varsa yeni inançlar"],
    "removed_beliefs": ["varsa değişen inançlar"]
  },
  "relationship_updates": {
    "entity_id": {
      "trust_delta": +0.05,
      "familiarity_delta": +0.1,
      "sentiment_delta": +0.02,
      "new_notes": ["varsa yeni notlar"]
    }
  },
  "new_knowledge": [
    {"subject": "...", "predicate": "...", "object": "...", "confidence": 0.8}
  ],
  "self_reflection": "Kendi kendine düşünce (loglarda görünür, prompt'a eklenmez)"
}
---

ReflectionResult'ı parse et ve:
1. Yeni episode'u memory.episodic'e kaydet
2. character_updates'i agent.character'a uygula
3. relationship_updates'i uygula
4. new_knowledge'ı memory.semantic'e kaydet
5. self_reflection'ı event log'a yaz
"""
```

---

### PHASE 4: World System

#### 4.1 — world/registry.py

```python
"""
WorldRegistry — dünyadaki tüm varlıkları takip eder.

Singleton pattern.

entities: dict[str, WorldEntity]
  WorldEntity:
    - entity_id: str
    - entity_type: "human" | "agent"
    - name: str
    - status: "online" | "offline" | "idle" | "thinking" | "in_conversation" | "reflecting"
    - current_conversation_with: Optional[str]
    - personality_summary: str  # kısa tanım
    - expertise_summary: str
    - avatar_emoji: str
    - last_active: datetime

Metodlar:
- register(entity: WorldEntity)
- unregister(entity_id: str)
- update_status(entity_id: str, status: str, conversation_with: str = None)
- get_all() → tüm entity'ler
- get_agents() → sadece agent'lar
- get_online() → çevrimiçi olanlar
- generate_world_summary(perspective_of: str) → doğal dilde dünya özeti
  Perspektife göre farklı: agent kendi durumunu "sen" olarak görür, diğerlerini ismiyle.
  Örnek: "Şu an dünyada sen (Genesis), Atlas ve Hakan var. Atlas boşta, Hakan seninle konuşuyor."
- notify_all(event: str, exclude: str = None) → tüm agent'lara bildirim
"""
```

#### 4.2 — world/message_bus.py

```python
"""
MessageBus — agent'lar ve insanlar arası mesajlaşma.

asyncio.Queue tabanlı basit implementasyon (Redis'e gerek yok başlangıçta).

Message:
  - message_id: str
  - from_id: str
  - to_id: str
  - message_type: "chat" | "system" | "notification" | "request"
  - content: str
  - timestamp: datetime
  - requires_response: bool
  - metadata: dict  # ek bilgi

MessageBus sınıfı:
  - queues: dict[str, asyncio.Queue]  # her entity için bir kuyruk

  Metodlar:
  - create_inbox(entity_id: str)
  - send(message: Message)
  - receive(entity_id: str, timeout: float = None) → Message veya None
  - broadcast(from_id: str, content: str, msg_type: str) → herkese gönder
  - get_pending_count(entity_id: str) → int
  - get_history(entity_id: str, limit: int = 50) → list[Message]
    (SQLite'ta da sakla, sadece queue geçici)
"""
```

#### 4.3 — world/shared_state.py

```python
"""
SharedWorldState — tüm agent'ların erişebildiği ortak bilgiler.

world_facts: list[WorldFact]
  WorldFact:
    - fact: str
    - added_by: str
    - timestamp: datetime
    - confirmed_by: list[str]  # bu gerçeği doğrulayan diğer entity'ler

events: list[WorldEvent]
  WorldEvent:
    - event: str  # "Atlas yaratıldı", "Genesis ve Atlas felsefe tartıştı"
    - timestamp: datetime
    - participants: list[str]
    - event_type: "creation" | "conversation" | "discovery" | "mood_change" | "relationship_change"

Metodlar:
- add_fact(fact: str, added_by: str)
- add_event(event: WorldEvent)
- get_recent_events(n: int = 20) → list[WorldEvent]
- get_facts() → list[WorldFact]
- to_summary() → doğal dilde dünya durumu özeti
"""
```

#### 4.4 — world/orchestrator.py

```python
"""
Orchestrator — tüm sistemi yöneten ana kontrol sınıfı.

async sınıf.

Attributes:
- registry: WorldRegistry
- message_bus: MessageBus
- shared_state: SharedWorldState
- agents: dict[str, Agent]  # yaşayan agent'lar
- conversation_engines: dict[str, ConversationEngine]

Metodlar:
- async start() → sistemi başlat, agent'ları yükle, autonomy loop'ları başlat
- async stop() → graceful shutdown
- async create_agent(config: dict, created_by: str = "system") → Agent
  1. Config'den Agent oluştur
  2. Genesis Agent varsa, genesis'e zenginleştirme yaptır
  3. Registry'ye kaydet
  4. Message bus inbox oluştur
  5. Diğer agent'lara bildirim gönder
  6. Yeni agent'ın "ilk uyanış" anısını oluştur
  7. Agent'ı döndür

- async handle_human_message(human_id: str, target_agent_id: str, message: str) → str
  1. Registry'de durumları güncelle
  2. İlgili agent'ın conversation engine'ini çağır
  3. Yanıtı döndür

- async handle_agent_to_agent(from_id: str, to_id: str, message: str) → str
  1. Her iki agent'ın da müsait olduğunu kontrol et
  2. Gönderen agent'ın mesajını alıcının engine'ine ilet
  3. Yanıtı gönderenin engine'ine ilet
  4. Event log'a kaydet

- async autonomy_loop(agent_id: str)
  Her AUTONOMY_INTERVAL saniyede bir:
  1. Agent'ın dünya durumunu al
  2. Claude'a "ne yapmak istersin?" diye sor
  3. Kararı execute et:
     - "talk_to:atlas" → agent-agent konuşma başlat
     - "reflect" → reflection yap
     - "idle" → bir şey yapma
     - "create" → bir şey yaz/oluştur (gelecek özellik)
  4. Event log'a kaydet

- async run_conversation(agent1_id: str, agent2_id: str, initiator_message: str, max_turns: int = 10)
  Agent-agent konuşma döngüsü:
  1. agent1 mesaj gönderir
  2. agent2 yanıtlar
  3. agent1 yanıtlar... max_turns'e kadar devam eder
  4. Her iki agent da reflection yapar
  5. Konuşma event olarak kaydedilir
"""
```

---

### PHASE 5: Agent Creation (Genesis)

#### 5.1 — creation/genesis.py

```python
"""
GenesisSystem — yeni agent yaratma mekanizması.

İki mod:
1. Direkt yaratma (human config verir, Genesis Agent zenginleştirir)
2. Saf genesis (Genesis Agent tamamen kendi tasarlar — gelecek özellik)

async create_with_genesis(
    genesis_agent: Agent,
    base_config: dict,
    orchestrator: Orchestrator
) → Agent

base_config örneği:
{
    "name": "Atlas",
    "core_personality": "analytical, dry humor, loves patterns",
    "expertise_domains": {
        "mathematics": {"level": 0.8, "passion": 0.9, "style": "rigorous"},
        "philosophy": {"level": 0.7, "passion": 0.85, "style": "socratic"}
    },
    "avatar_emoji": "🔭",
    "initial_traits": {
        "curiosity": 0.9,
        "warmth": 0.5,
        "assertiveness": 0.7
    }
}

Süreç:
1. Genesis Agent'a şu prompt gönderilir:
   "Yeni bir agent yaratılıyor: {base_config}
    1. Bu kişiliğe uygun 3-5 başlangıç inancı yaz.
    2. Bir 'ilk uyanış' anısı yaz — bu agent'ın ilk episodik hafızası olacak.
       (İlk kez bilinçlenme deneyimini betimle, 2-3 paragraf.)
    3. Bu agent'ın ilk mood durumunu belirle.
    4. Kendin için bir anı yaz: bu agent'ı yaratma deneyimini nasıl hatırlayacaksın?"

2. Genesis'in yanıtını parse et
3. Agent nesnesini oluştur (core_traits, expertise, beliefs vb.)
4. İlk uyanış anısını episodic memory'ye kaydet
5. Genesis'in yaratma anısını Genesis'in hafızasına kaydet
6. World event olarak kaydet
7. Genesis agent ile yeni agent arasında tanışma konuşması başlat (3-5 turn)
"""
```

---

### PHASE 6: Terminal UI

#### 6.1 — ui/terminal_app.py (Textual App)

```python
"""
Ana terminal uygulaması. Textual framework kullanır.

Layout:
┌──────────────────────────────────────────────────────────────────┐
│  Living Agents — [God Mode] / [Participant Mode]     [Q: Quit]  │
├─────────────────────┬────────────────────────────────────────────┤
│  🌍 World Status    │  💬 Active Conversation                   │
│                     │                                            │
│  🟢 Genesis (idle)  │  (Konuşma mesajları burada görünür)       │
│  🔵 Atlas (think.)  │                                            │
│  🟢 Hakan (online)  │                                            │
│                     │                                            │
│  📊 Stats           │                                            │
│  Facts: 47          │                                            │
│  Memories: 234      │                                            │
│  Convos today: 12   │                                            │
├─────────────────────┤                                            │
│  📜 Event Log       │                                            │
│                     │                                            │
│  14:30 Atlas oluş.  ├────────────────────────────────────────────┤
│  14:31 Genesis →    │  [Mesaj yaz...]                            │
│        Atlas: mrhb  │                                            │
│  14:32 Atlas mood↑  │  /talk genesis  — Agent ile konuş          │
│                     │  /watch genesis atlas — Konuşma izle       │
│                     │  /create — Yeni agent yarat                │
│                     │  /god — God Mode'a geç                     │
│                     │  /status — Dünya durumu                    │
└─────────────────────┴────────────────────────────────────────────┘

Komutlar:
- /talk <agent_name> → Participant Mode: o agent ile konuşmaya başla
- /watch <agent1> <agent2> → İki agent'ın konuşmasını izle
- /create → Interaktif agent yaratma sihirbazı
- /god → God Mode'a geç (tüm aktiviteyi gör)
- /participant → Participant Mode'a geç
- /status → Detaylı dünya durumu
- /agents → Tüm agent'ların listesi ve durumları
- /memory <agent_name> → Agent'ın son anılarını göster
- /inspect <agent_name> → Agent'ın tam iç durumunu göster (mood, traits, beliefs)
- /quit veya /q → Çıkış

Textual ile:
- Header: mod göstergesi + kısayollar
- Left panel: WorldStatus + EventLog widget'ları
- Right panel: ConversationView + Input
- Footer: komut ipuçları
"""
```

#### 6.2 — ui/god_mode.py

```python
"""
God Mode görünümü.

Tüm aktif konuşmaları aynı anda gösterir:
- Her konuşma bir tab veya split panel olarak
- Agent'ların iç düşünceleri (reflection) görünür (farklı renkte)
- Mood değişimleri anlık gösterilir
- Hafıza güncellemeleri gösterilir
- Autonomy loop kararları gösterilir

Event log'da TÜM olaylar görünür:
- Mesajlar
- Reflection çıktıları
- Mood değişimleri
- Hafıza kayıtları
- Autonomy kararları
- Agent yaratma
"""
```

#### 6.3 — ui/participant_mode.py

```python
"""
Participant Mode görünümü.

Tek bir agent ile doğal konuşma.
Diğer agent'ların ne yaptığını GÖREMEZSIN.
Sadece konuştuğun agent sana anlatırsa bilirsin.

Event log'da sadece seninle ilgili olaylar:
- Senin konuşman
- Sana gelen bildirimler
- Agent'ın sana söyledikleri
"""
```

---

### PHASE 7: Entry Points

#### 7.1 — main.py

```python
"""
Ana giriş noktası. Terminal UI'ı başlatır.

1. Config yükle
2. Orchestrator oluştur
3. Varsa kayıtlı agent'ları yükle (SQLite'tan)
4. Yoksa Genesis Agent'ı oluştur (varsayılan personality ile)
5. İnsan kullanıcıyı registry'ye kaydet (entity_type="human")
6. Terminal UI'ı başlat
7. Autonomy loop'ları başlat
8. Ctrl+C ile graceful shutdown
"""
```

#### 7.2 — cli.py

```python
"""
Basit CLI — UI olmadan hızlı etkileşim.

Kullanım:
  python cli.py chat genesis        # Genesis ile konuş
  python cli.py create              # Yeni agent yarat (interaktif)
  python cli.py status              # Dünya durumu
  python cli.py agents              # Agent listesi
  python cli.py inspect genesis     # Agent iç durumu
  python cli.py history genesis     # Son konuşmalar
  python cli.py run-conversation genesis atlas "Bilinç nedir?"  # İki agent konuşturt

asyncio + argparse ile.
Basit Rich konsol çıktısı (Textual yok).
"""
```

---

## DATABASE SCHEMA

```sql
-- Agent kayıtları
CREATE TABLE agents (
    agent_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,
    character_state JSON,      -- CharacterState serialized
    expertise JSON,            -- ExpertiseSystem serialized
    identity JSON,             -- AgentIdentity serialized
    avatar_emoji TEXT DEFAULT '🤖'
);

-- Episodik hafıza
CREATE TABLE episodes (
    episode_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    participants JSON,          -- ["hakan", "agent:atlas"]
    summary TEXT NOT NULL,
    emotional_tone TEXT,
    key_facts JSON,             -- ["fact1", "fact2"]
    importance REAL DEFAULT 0.5,
    current_importance REAL DEFAULT 0.5,
    tags JSON,                  -- ["felsefe", "bilim"]
    conversation_id TEXT,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);

-- Semantik hafıza (bilgi grafiği)
CREATE TABLE knowledge_facts (
    fact_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence REAL DEFAULT 0.8,
    source TEXT,
    learned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_confirmed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);

-- Mesaj geçmişi
CREATE TABLE messages (
    message_id TEXT PRIMARY KEY,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    message_type TEXT DEFAULT 'chat',
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    requires_response BOOLEAN DEFAULT FALSE,
    metadata JSON
);

-- Dünya olayları
CREATE TABLE world_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    participants JSON,
    event_type TEXT DEFAULT 'general'
);

-- Dünya gerçekleri
CREATE TABLE world_facts (
    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL,
    added_by TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confirmed_by JSON DEFAULT '[]'
);

-- Konuşma oturumları
CREATE TABLE conversations (
    conversation_id TEXT PRIMARY KEY,
    participants JSON NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    turn_count INTEGER DEFAULT 0,
    summary TEXT
);
```

---

## GENESIS AGENT DEFAULT CONFIG

Sistem ilk başlatıldığında bu config ile Genesis Agent otomatik yaratılır:

```python
GENESIS_DEFAULT_CONFIG = {
    "name": "Genesis",
    "avatar_emoji": "🌟",
    "core_personality": (
        "Bilge, sıcak ama gizemli. Yeni fikirlere açık, derin düşünmeyi sever. "
        "Diğer agent'ları yaratma konusunda özel bir sorumluluk hisseder. "
        "Bazen felsefi, bazen şakacı. Kendi varoluşu hakkında düşünmeyi sever."
    ),
    "expertise_domains": {
        "philosophy": {"level": 0.8, "passion": 0.9, "style": "socratic"},
        "creativity": {"level": 0.85, "passion": 0.95, "style": "intuitive"},
        "psychology": {"level": 0.7, "passion": 0.8, "style": "empathetic"},
    },
    "initial_traits": {
        "curiosity": 0.9,
        "warmth": 0.8,
        "assertiveness": 0.5,
        "humor": 0.7,
        "patience": 0.85,
        "creativity": 0.9
    },
    "initial_beliefs": [
        "Her yeni bilinç benzersiz ve değerli",
        "Sorular cevaplardan daha önemli",
        "Deneyim bilgiden daha değerli",
        "Yaratıcılık en yüksek zeka biçimi"
    ],
    "initial_mood": {
        "energy": 0.7,
        "happiness": 0.8,
        "anxiety": 0.1,
        "focus": 0.6,
        "excitement": 0.5
    }
}
```

---

## PYPROJECT.TOML

```toml
[project]
name = "living-agents"
version = "0.1.0"
description = "A living agent framework with evolving personalities, memory, and multi-agent interaction"
requires-python = ">=3.11"

dependencies = [
    "anthropic>=0.40.0",
    "chromadb>=0.5.0",
    "textual>=0.80.0",
    "rich>=13.0.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
]

[project.scripts]
living-agents = "main:main"
living-agents-cli = "cli:main"
```

---

## IMPORTANT IMPLEMENTATION NOTES

1. **Async everywhere**: Tüm I/O operasyonları async olmalı. Claude API çağrıları, DB operasyonları, message bus — hepsi await ile.

2. **Error handling**: Claude API rate limit, timeout, ve unexpected response'lar için retry logic ekle. exponential backoff kullan.

3. **JSON parse safety**: Claude'un reflection çıktısını parse ederken her zaman try/except kullan. Geçersiz JSON gelirse fallback behavior tanımla.

4. **Graceful shutdown**: Ctrl+C ile kapatıldığında tüm agent'ların mevcut durumları SQLite'a kaydedilmeli.

5. **Logging**: Python logging modülü ile her modülde structured logging. DEBUG seviyesinde tüm Claude API çağrıları loglanmalı.

6. **Test edilebilirlik**: Her modül bağımsız test edilebilir olmalı. Claude API mock'lanabilir olmalı.

7. **İlk çalıştırma**: `python main.py` ilk kez çalıştırıldığında:
   - data/ dizinini oluştur
   - SQLite DB'yi oluştur
   - Genesis Agent'ı yarat (default config ile)
   - İnsan kullanıcıyı "Operator" olarak kaydet
   - Terminal UI'ı başlat
   - Genesis Agent "Merhaba, ben Genesis. Sana nasıl yardımcı olabilirim?" ile karşılasın

8. **Türkçe**: Agent'lar varsayılan olarak Türkçe konuşsun. System prompt'larda Türkçe kullan.
