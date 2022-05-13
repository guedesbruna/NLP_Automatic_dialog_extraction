{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "e943a33b",
   "metadata": {},
   "source": [
    "# Preprocessing\n",
    "\n",
    "######################################### \n",
    "## All preprocessing just needs to be done once. \n",
    "### If already done go direclty to DA patterns to load csv\n",
    "######################################### "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "de9d64ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from pandas.io.json import json_normalize\n",
    "import re\n",
    "\n",
    "# os.listdir('./data_TM2')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "03ccba6b",
   "metadata": {},
   "source": [
    "#### Open all datasets from different tasks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "92e7d0d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_t1 = pd.read_json('./data_TM2/flights.json')\n",
    "df_t2 = pd.read_json('./data_TM2/food-ordering.json')\n",
    "df_t3 = pd.read_json('./data_TM2/hotels.json')\n",
    "df_t4 = pd.read_json('./data_TM2/movies.json')\n",
    "df_t5 = pd.read_json('./data_TM2/music.json')\n",
    "df_t6 = pd.read_json('./data_TM2/restaurant-search.json')\n",
    "df_t7 = pd.read_json('./data_TM2/sports.json')\n",
    "\n",
    "# create extra column in the beginning for type of task before merging\n",
    "df_t1.insert(0, 'task', 'flights')\n",
    "df_t2.insert(0, 'task', 'food-ordering')\n",
    "df_t3.insert(0, 'task', 'hotels')\n",
    "df_t4.insert(0, 'task', 'movies')\n",
    "df_t5.insert(0, 'task', 'music')\n",
    "df_t6.insert(0, 'task', 'restaurant-search')\n",
    "df_t7.insert(0, 'task', 'sports')\n",
    "\n",
    "# merged file created is giving errors in process_raw function\n",
    "# # merge dataframes\n",
    "# frames = [df_t1, df_t2, df_t3, df_t4, df_t5, df_t6, df_t7]\n",
    "# all_tasks_df = pd.concat(frames, ignore_index=True)\n",
    "\n",
    "# # saving as csv\n",
    "# all_tasks_df.to_csv('./data_TM2/all_tasks.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7f7cfaa8",
   "metadata": {},
   "source": [
    "#### Normalize json and change to pandas "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "867f1f88",
   "metadata": {},
   "outputs": [],
   "source": [
    "def process_raw(df): #, name):\n",
    "    dialogs = []\n",
    "\n",
    "    for e,i in enumerate(df['utterances']):\n",
    "        for j in i:\n",
    "            new_df=pd.json_normalize(j)\n",
    "            new_df.insert(0, 'task', df['task'][e])\n",
    "            new_df.insert(1, 'conversation_id', df['conversation_id'][e])\n",
    "            new_df.insert(2, 'instruction_id', df['instruction_id'][e])\n",
    "        \n",
    "            dialogs.append(new_df)\n",
    "\n",
    "    large_df = pd.concat(dialogs, ignore_index=True)\n",
    "#   large_df.to_csv('./data_TM2/processed_utterances_'+name+'.csv')\n",
    "    \n",
    "    return large_df"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "43582db7",
   "metadata": {},
   "source": [
    "#### Create unique dataset with normalized tasks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "607586e6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def unique(df_list):\n",
    "    large_frames = []\n",
    "\n",
    "    for df_ in df_list:\n",
    "        large_df = process_raw(df_)\n",
    "        large_frames.append(large_df)\n",
    "\n",
    "    all_tasks = pd.concat(large_frames, ignore_index=True)\n",
    "    all_tasks.to_csv('./data_TM2/processed_utterances_all_tasks_TESTE2.csv')\n",
    "    \n",
    "    return all_tasks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "03061b66",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_list = [df_t1, df_t2, df_t3, df_t4, df_t5, df_t6, df_t7]\n",
    "all_tasks = unique(df_list)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9b9b2f80",
   "metadata": {},
   "source": [
    "## Load whole dataset to work with DA patterns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c635da6d",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('./data_TM2/processed_utterances_all_tasks_TESTE2.csv', index_col=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "917a93f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# new_text = []\n",
    "# for ut in df['text']:\n",
    "#     new_ut = ut.split('. ')\n",
    "#     new_text.append(new_ut)\n",
    "\n",
    "# df['new_text'] = new_text\n",
    "\n",
    "\n",
    "# #possibly insert here new part\n",
    "\n",
    "\n",
    "# df = df.explode('new_text')\n",
    "# df = df.reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9cb3bd85",
   "metadata": {},
   "outputs": [],
   "source": [
    "# #essa parte é de antes porem com split incluindo simbolo do split\n",
    "new_text = []\n",
    "for ut in df['text']:\n",
    "    new_ut = re.split('(\\.)', ut)\n",
    "    new_text.append(new_ut)\n",
    "df['new_text'] = new_text\n",
    "\n",
    "#to keep delimiter in sentence\n",
    "#not very efficient way of doing it, apparently a better way using regex symbols\n",
    "\n",
    "new_lista = []\n",
    "for row in df['new_text']:\n",
    "#     print(row)\n",
    "    for e, word in enumerate(row):\n",
    "        if word == '.':\n",
    "            new_lista.append(row[e-1]+row[e])\n",
    "            new_lista.remove(row[e-1])\n",
    "        else:\n",
    "            new_lista.append(word)\n",
    "\n",
    "df = df.explode('new_text')\n",
    "df = df.reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b8b7d4de",
   "metadata": {},
   "source": [
    "### Find substrings\n",
    "make list of words and rules for labeling DA\n",
    "https://towardsdatascience.com/check-for-a-substring-in-a-pandas-dataframe-column-4b949f64852#:~:text=Using%20%E2%80%9Ccontains%E2%80%9D%20to%20Find%20a,substring%20and%20False%20if%20not."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4920aabb",
   "metadata": {},
   "source": [
    "- add new column for subset. since each utterance can only be one thing one column for all DA should be enough. \n",
    "- Later I'll have to check the vocabulary used for generic rules (FS_base_greeting or whatever is)\n",
    "- Also study in the UX book what is repair and other patterns:\n",
    "    - page 243 to 246: NCF pattern language summary\n",
    "\n",
    "Tips:\n",
    "- #NOT_INCLUDE_WORD: \n",
    "    - not_fifa = football_soccer_games.loc[~football_soccer_games['Name'].str.contains('FIFA')]\n",
    "- #to make a new dataframe with slicing subset:\n",
    "    - new_df = df.loc[df['text'].str.contains('|'.join(end_of_request), case=False)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "51656f7e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# DA indicators. Obs: the vars bellow are sensitive to spacing !!! 'hi' != ' hi'\n",
    "\n",
    "greeting = [' hi','hi.','hello','bye'] #76942\n",
    "repair_initiator = ['sorry','repeat', 'understand', 'mean'] #1991\n",
    "#repair is what comes after the repair initiator\n",
    "\n",
    "request_summary = ['correct?','confirm'] #2841 #when you ask an information to be confirmed\n",
    "confirmation = ['yes','correct.','correct ', 'that\\'s it', 'indeed'] #confirmation é a resposta que confirma: ex yes | correct | that's it\n",
    "sentence_closer = ['awesome','great','perfect','exactly','that\\'s all', 'thanks', 'okay', 'thank you'] #18315\n",
    "question = ['\\?']\n",
    "receipt = ['You are welcome', 'you\\'re welcome']\n",
    "\n",
    "#choose if i'll deal with whole iterances or them split\n",
    "\n",
    "# temp=df.text.fillna(\"0\")\n",
    "# df['DA_indicators'] =  np.where(temp.str.contains('|'.join(repair_initiator), case=False), \"repair_initiator\",\n",
    "#                             np.where(temp.str.contains('|'.join(greeting), case=False), \"greeting\",\n",
    "#                             np.where(temp.str.contains('|'.join(confirmation), case=False),  \"confirmation\",\n",
    "#                             np.where(temp.str.contains('|'.join(question), case=False),  \"question\",\n",
    "#                             np.where(temp.str.contains('|'.join(sentence_closer), case=False), \"sentence_closer\", \"x\")))))\n",
    "\n",
    "temp2=df.new_text.fillna(\"0\")\n",
    "df['DA_indicators_sent'] =  np.where(temp2.str.contains('|'.join(repair_initiator), case=False), \"repair_initiator\",\n",
    "                            np.where(temp2.str.contains('|'.join(greeting), case=False), \"greeting\",\n",
    "                            np.where(temp2.str.contains('|'.join(confirmation), case=False),  \"confirmation\",\n",
    "                            np.where(temp2.str.contains('|'.join(question), case=False),  \"question\",\n",
    "                            np.where(temp2.str.contains('|'.join(sentence_closer), case=False), \"sentence_closer\", \"x\")))))\n",
    "\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb7961c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# #to keep delimiter in sentence\n",
    "# #not very efficient way of doing it\n",
    "# #apparently a better way using regex symbols\n",
    "\n",
    "# new_lista = []\n",
    "# for row in df['new_text'].iloc[0:50]:\n",
    "# #     print(row)\n",
    "#     for e, word in enumerate(row):\n",
    "#         if word == '.':\n",
    "#             new_lista.append(row[e-1]+row[e])\n",
    "#             new_lista.remove(row[e-1])\n",
    "#         else:\n",
    "#             new_lista.append(word)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aa7ea204",
   "metadata": {},
   "outputs": [],
   "source": [
    "#no pior dos casos faco um split de cada vez.\n",
    "#agora olhar como manter o simbolo usado pro split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e39ff06a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import re\n",
    "# testando = 'as, la? ha! el?'\n",
    "# lista = re.split('(\\?)', testando)\n",
    "# lista"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c068621",
   "metadata": {},
   "outputs": [],
   "source": [
    "# #to keep delimiter in sentence\n",
    "# #not very efficient way of doing it\n",
    "# #apparently a better way using regex symbols\n",
    "\n",
    "# new_lista = []\n",
    "# for e, word in enumerate(df['text']):\n",
    "#     if word == '?':\n",
    "#         print('hola')\n",
    "#         new_lista.append(lista[e-1]+lista[e])\n",
    "#         new_lista.remove(lista[e-1])\n",
    "#     else:\n",
    "#         new_lista.append(word)\n",
    "            \n",
    "\n",
    "# new_lista"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "90c4425f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# for e, word in enumerate(lista):\n",
    "#     if word == '?':\n",
    "#         frase = lista[e-1]+lista[e]\n",
    "# frase"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
