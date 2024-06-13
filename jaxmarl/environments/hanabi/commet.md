# 基本设定

+ 卡牌颜色：【红， 黄， 蓝，白，绿】

+ 每个颜色卡牌数量 【数字： 数量】

  ```json
  {
      1:3,
      2:2,
      3:2,
      4:2,
      5:1,
  }
  ```

+ info_tokens: 提示牌（上限8张）

+ life_tokens: 生命牌（上限3张）



# 玩家操作

+ 提供信息给下一位玩家, 告诉它某花色/数字的牌在什么位置(必须是正确的)：info_token - 1
+ 弃牌：弃牌至弃牌堆，此牌不能再加入游戏及打出：info_token + 1 (注意不能超过上限) 
  + 摸一张牌
+ 打牌：若此牌可按花色顺序正确排列，则有效，否则消耗一张生命牌（life_token - 1）
  + 摸一张牌（不管打出的牌有没有成功接上）



# 被动触发

+ 当玩家成功打出一种花色的完整一组5张牌，如：红色 12345， info_token + 1 (注意不能超过上限)



# 结束条件

+ 牌库没有牌的时候
+ 25分全拿满
+ life_token 全消耗完





# class State 变量

```python
	# 抽牌堆
    deck: chex.Array # after shuffling the card, 50 card deck with 50x5x5 size
    # 弃牌堆，同上，初始值都是0，因为开始时没有弃牌
    discard_pile: chex.Array
    
    # 目前出牌区的状态：5*5的数组，初始值为0,假设红色牌出到第三个 则代表红色牌的行或者列前三个都要置为1
    fireworks: chex.Array
    
    
    player_hands: chex.Array# cards currently hold by each agent agentNum*handsize*5*5
    
    # 同life token
    info_tokens: chex.Array
    terminal: bool
    
    # 3*1 的数组， 1代表lifetoken还可以被使用
    life_tokens: chex.Array
    
    # card_knowledge 初始值被置为1,为什么为1不是0 维度： num_agents(2)*hand_size(5)* (num_colors(5)*num_ranks(5)) 所有玩家对于自己卡牌的认知
    card_knowledge: chex.Array
    
    #colors_revealed 初始值为0， 维度：num_agents(2) * hand_size(5) * num_colors(5)
    #假设对于第一个agent 已经告诉他第一张牌是红色，则（0，0，0）的位置变为1， 假设第0位代表红色
    colors_revealed: chex.Array
    
    # 同上
    ranks_revealed: chex.Array
    num_cards_dealt: int # cards num that has been chosen from the deck
    
    # 弃牌堆数量
    num_cards_discarded: int
    
    # agent_num* num_moves 上一轮玩家们各干了什么
    last_moves: chex.Array#Records the last moves made by players. Useful for keeping track of the game history or for AI to analyze past actions.
    
    # 目前轮到哪个agent了，agent_num(2)*1的数组，如果是当前agent，对应的index位置置1
    cur_player_idx: chex.Array
    
    out_of_lives: bool#A boolean flag that becomes true if the players lose all their life tokens.
    
    # 上一轮是第几轮
    last_round_count: int
    
    # ？？？
    bombed: bool
    
    #一个50*1的数组，0表示被抽走，最下面的牌是0，最上面的牌是49
    remaining_deck_size: chex.Array

```









# HanabiGame: Class变量定义

```python
num_agents=2 # 游戏人数
num_colors=5 # 卡牌颜色数量
num_ranks=5 # 数字有5种
hand_size=5 # 手里最多拿5张牌
max_info_tokens=8 # 提示牌最多8张
max_life_tokens=3 # 生命牌最多3张
num_cards_of_rank=np.array([3, 2, 2, 2, 1]) # 每种数字有几张牌，每个颜色都有3个1...
agents=None # init： ['agent_0', 'agent_1']
action_spaces=None
observation_spaces=None
obs_size=None
num_moves=None # 表示游戏中定义的所有可能动作的数量
agent_range = jnp.arange(num_agents) # [0, 1] 玩家范围
deck_size = jnp.sum(num_cards_of_rank) * num_colors # 牌堆数量



num_moves = np.sum(np.array([
                # noop
                1,
                # discard, play (弃哪张牌，或打哪张牌)
                # 编码： 0-4 弃牌， 5-9 打牌，之后的是hint
                hand_size * 2,
                # hint color, rank（提示哪一位玩家，那个数字或者颜色）
                (num_agents - 1) * (num_colors + num_ranks)
            ])).squeeze() # 所有玩家在一个回合内所有可能的行动数量（1+10+1*10=21）


obs_size = (
                    # card knowledge 手牌数量*每张牌的可能性
    				# 每个player对于自己手牌的knowledge， 5*cross join of colr and rank
                    (hand_size * (num_colors * num_ranks)) +
                    # color and rank hints 手牌数量*每张手牌接收到的颜色和级别提示的可能性
    				# 每个player对于自己手牌接受到的颜色或者rank提示 5x（5+5）
                    (hand_size * (num_colors + num_ranks)) +
                    # other hands 看到的所有其他玩家的手牌信息
                    # 除自己以外的玩家数量 * 手牌张数 * 可能的颜色 * 可能的数字
                    ((num_agents - 1) * hand_size * num_colors * num_ranks) +
                    # available actions
                    # self.num_moves +
                    # fireworks
                    # 玩家看到的现在出牌区的状态：哪张牌出到了数字几
                    (num_colors * num_ranks) +
                    # info tokens
                    # 可用的info token数量
                    max_info_tokens +
                    # life tokens
                    # 可用的life token数量
                    max_life_tokens +
                    # last moves
                    # 所有玩家可能的所有动作数量（包括自己的）
                    (num_agents * self.num_moves) +
                    # current player
                    # 当前是哪位玩家在玩
                    num_agents +
                    # discard pile
                    # 弃牌堆的数量(最多50张)*牌的信息（5*5）
                    # （每个颜色牌的总数10）*颜色数量5 *牌的信息（5*5）
                    np.sum(num_cards_of_rank) * num_colors * num_colors * num_ranks +
                    # remaining deck size
                    # 剩余牌堆的情况（只能看到数量，看不到信息）
                    np.sum(num_cards_of_rank) * num_colors
            )

# 观测的维度大小

```

![image-20240201132038339](https://markdown-1301334775.cos.eu-frankfurt.myqcloud.com/image-20240201132038339.png)

# ENV

+ state：上帝视角
+ obs：agent可以看到的范围
+ reward：



## Reset

+ 初始化state --> 初始化obs





## Step_env

+ 一轮之后（每个agent轮流动一次），更新环境的state，返回state，返回所有agent的obs，返回done（游戏是否结束）

## Step_agent(state, agentid, 哪个action)

+ 更新现在环境
+ 根据采取的action
+ 返回state和reward



## Get_legal_move

+ 根据现在state判断，返回下一步所有可行的操作



## get_obs()

+ 根据当前的状态，判断每个agent可以看到什么,主要是根据State得到每个agent可以看到的obs， linearize成一个1680x1的数组 最后返回一个dict





## obs_size

+ 我已知的自己的手牌信息
+ 我可能接受到其他玩家的卡牌信息
+ 其他玩家的手牌信息
+ 现在出牌区的情况
+ 可用的info和life token数量
+ 所有玩家可能的所有动作数量
+ 当前是哪位玩家在玩

```python
        if action_spaces is None: # dict，全局唯一，表示某个agent当前选择的action
            self.action_spaces = {i: Discrete(self.num_moves) for i in self.agents}
        if observation_spaces is None: # 在2的1680次方种情况中选一种？？看一下agent到底是不是通过obs_space来决定下一步干什么
            self.observation_spaces = {i: Discrete(self.obs_size) for i in self.agents}
            
            
            # 暂时就理解为action_spaces = 21， observation_spaces = 1680
            # 需要1680个bit 才能表示obs_space （可以多位为1）
            # 需要21个bit 才能表示action_space (但是只能一个为1)
            
```



#### hanabi.py

```python
line: 183
# 表示颜色和数字组合 50*2 
# cross join [01234] and [0001122334]
color_rank_pairs = jnp.dstack(jnp.meshgrid(colors, ranks)).reshape(-1, 2)

line 186
# 打乱数组的顺序
shuffled_pairs = jax.random.permutation(_key, color_rank_pairs, axis=0)


line 188
# _gen_cards(axid， unused) 0是axid的初始值 decksize是函数一共需要迭代运行的次数 None是初始数组
# deck相当于50个5*5的二维数组，每个二维数组只有一格为1，代表这张牌是某颜色*数字
_, deck = lax.scan(_gen_cards, 0, None, self.deck_size)
# 这个方法只是为了把shuffled_pairs变成值只有0，1的数据结构

# 初始抽牌阶段： num_agents* handsize个牌的数量，各位玩家各抽到了什么样的牌
# 每张牌 5*5 ，所以num_agents* handsize * num_colors * num_ranks
 _, hands = lax.scan(_deal_cards, 0, None, self.num_agents)
    
    
    
num_cards_dealt = self.num_agents * self.hand_size# 抽出牌的数量


line 211：
# 抽了十张牌，把0-9全部置为 5x5的0矩阵 代表已经被抽走了。 Note：最上面的牌序号为0 最下面的序号为49
deck = deck.at[:num_cards_dealt].set(jnp.zeros((self.num_colors, self.num_ranks)))
```

