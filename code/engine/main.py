import engine


# MAIN LOOP
def main():
    game = engine.Game()
    # init()
    game.init_round()

    while game.run:
        game.get_delta_time()
        game.check_for_events()
        # -------------------------------------------------------
        # AI INPUT (intern)
        # -------------------------------------------------------
        game.update()
        game.draw()
        game.check_state()
        # -------------------------------------------------------
        #  AI outpput
        # -------------------------------------------------------

if __name__ == "__main__":
    main()