from src.basket_manager import BasketManager


def test_add_item():
    basket = BasketManager()

    added = basket.add_item(101, "Coca-Cola")

    assert added is True
    assert basket.get_summary()["total_items"] == 1
    assert basket.get_summary()["basket_summary"] == {
        "Coca-Cola": 1
    }


def test_duplicate_track_id_is_not_counted_twice():
    basket = BasketManager()

    basket.add_item(101, "Coca-Cola")
    added = basket.add_item(101, "Coca-Cola")

    assert added is False
    assert basket.get_summary()["total_items"] == 1


def test_multiple_products_are_counted_correctly():
    basket = BasketManager()

    basket.add_item(101, "Coca-Cola")
    basket.add_item(102, "Pepsi")
    basket.add_item(103, "Coca-Cola")

    summary = basket.get_summary()

    assert summary["total_items"] == 3
    assert summary["basket_summary"] == {
        "Coca-Cola": 2,
        "Pepsi": 1
    }


def test_remove_item():
    basket = BasketManager()

    basket.add_item(101, "Coca-Cola")
    removed = basket.remove_item(101)

    assert removed is True
    assert basket.get_summary()["total_items"] == 0
    assert basket.get_summary()["basket_summary"] == {}


def test_remove_unknown_item():
    basket = BasketManager()

    removed = basket.remove_item(999)

    assert removed is False


def test_reset_basket():
    basket = BasketManager()

    basket.add_item(101, "Coca-Cola")
    basket.add_item(102, "Pepsi")

    basket.reset()

    assert basket.get_summary()["total_items"] == 0
    assert basket.get_summary()["basket_summary"] == {}
