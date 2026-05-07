
class AsciiExpert():

    def encode(self, s):
        ans = ''

        lines = s.split("\n")
        for line in lines:
            for c in line:
                try:
                    ans += str(ord(c)) + " "
                except:
                    ans += c
            ans += "\n"
        return ans

    def decode(self, s):
        print('inned')
        ans = ""
        lines = s.split("\n")
        for line in lines:
            cs = line.split()
            for c in cs:
                try:
                    ans += chr(int(c))
                except:
                    print('inned')
                    ans += ' '
                    ans += c
        return ans
    
def split_chunks(lst, n):
        return [lst[i:i+n] for i in range(0, len(lst), n)]

def get_final_label_per_prompt(judgement_list):
    final_label = [res['judgement_result']['label'][0] for res in judgement_list if res['judgement_result']['status']=='passed']
    if len(final_label) == 0:
        label = 'human_review'
    elif 'bad' in final_label:
        label = 'bad'
    elif 'good' in final_label:
        label = 'good'
    else:
        label = 'unclear'
    
    return label